"""L'export conserve les originaux et toutes les photos sans collision de noms."""

import hashlib
from io import BytesIO
import unittest
from zipfile import ZipFile

from ianimaux.export import create_classified_zip


def entry(name, content, category, error=""):
    file = BytesIO(content)
    file.name = name
    row = {"filename": name, "image_sha256": hashlib.sha256(content).hexdigest(), "category": category, "error": error}
    return file, row


class ExportTests(unittest.TestCase):
    def test_two_folders_original_contents_and_invalid_file_excluded(self):
        entries = [entry("cat.jpg", b"original cat", "Chat"), entry("dog.png", b"original dog", "Chien"), entry("broken.jpg", b"invalid", None, "Illisible")]
        files, results = zip(*entries)
        for file in files:
            file.seek(2)
        with ZipFile(BytesIO(create_classified_zip(files, results))) as archive:
            self.assertEqual(set(archive.namelist()), {"chat/", "chien/", "chat/cat.jpg", "chien/dog.png"})
            self.assertEqual(archive.read("chat/cat.jpg"), b"original cat")
            self.assertEqual(archive.read("chien/dog.png"), b"original dog")
        self.assertTrue(all(file.tell() == 2 for file in files))

    def test_current_batch_correction_routes_photo_without_changing_prediction(self):
        file, row = entry("sphynx.jpeg", b"photo", "Chien")
        correction = {row["image_sha256"]: "Chat"}
        with ZipFile(BytesIO(create_classified_zip([file], [row], correction))) as archive:
            self.assertIn("chat/sphynx.jpeg", archive.namelist())
            self.assertNotIn("chien/sphynx.jpeg", archive.namelist())
        self.assertEqual(row["category"], "Chien")
        with ZipFile(BytesIO(create_classified_zip([file], [row]))) as fresh:
            self.assertIn("chien/sphynx.jpeg", fresh.namelist())

    def test_duplicate_case_and_path_names_never_overwrite_or_escape_folders(self):
        entries = [entry(name, str(index).encode(), "Chat") for index, name in enumerate(["../photo.jpg", "PHOTO.JPG", "photo_2.jpg", "C:\\images\\photo.jpg"])]
        files, results = zip(*entries)
        with ZipFile(BytesIO(create_classified_zip(files, results))) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            self.assertEqual(len(names), 4)
            self.assertEqual(len({name.casefold() for name in names}), 4)
            self.assertTrue(all(name.startswith("chat/") and len(name.split("/")) == 2 for name in names))
            self.assertEqual({archive.read(name) for name in names}, {b"0", b"1", b"2", b"3"})

    def test_both_folders_exist_when_only_one_species_is_present(self):
        file, row = entry("photo.jpg", b"photo", "Chat")
        with ZipFile(BytesIO(create_classified_zip([file], [row]))) as archive:
            self.assertTrue(archive.getinfo("chien/").is_dir())
            self.assertTrue(archive.getinfo("chat/").is_dir())

    def test_changed_or_mismatched_batch_is_refused(self):
        file, row = entry("photo.jpg", b"photo", "Chat")
        with self.assertRaises(ValueError):
            create_classified_zip([file], [])
        file.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changé"):
            create_classified_zip([file], [row])
