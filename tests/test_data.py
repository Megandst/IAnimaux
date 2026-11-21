"""Contrôles d'intégrité sur de petites données artificielles, jamais des métriques IA."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from ianimaux.data import find_exact_duplicates, inspect_image, read_annotations


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "annotations").mkdir()

    def write_annotations(self, trainval, test):
        (self.root / "annotations" / "trainval.txt").write_text(trainval)
        (self.root / "annotations" / "test.txt").write_text(test)

    def test_species_comes_from_annotation_not_filename(self):
        self.write_annotations("Odd_name_1 1 2 1\n", "other_breed_1 2 1 1\n")
        frame = read_annotations(self.root)
        self.assertEqual(frame.species.tolist(), ["Chien", "Chat"])
        self.assertEqual(frame.target.tolist(), [1, 0])

    def test_same_identifier_in_train_and_test_is_rejected(self):
        self.write_annotations("Cat_1 1 1 1\n", "Cat_1 1 1 1\n")
        with self.assertRaisesRegex(ValueError, "Identifiants"):
            read_annotations(self.root)

    def test_conflicting_species_for_same_class_is_rejected(self):
        self.write_annotations("Pet_1 1 1 1\n", "Pet_2 1 2 1\n")
        with self.assertRaisesRegex(ValueError, "incohérente"):
            read_annotations(self.root)

    def test_invalid_species_is_rejected(self):
        self.write_annotations("Pet_1 1 3 1\n", "Other_1 2 2 1\n")
        with self.assertRaisesRegex(ValueError, "Étiquette"):
            read_annotations(self.root)

    def test_corrupt_and_missing_images_are_reported(self):
        corrupt = self.root / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        for path in (corrupt, self.root / "missing.jpg"):
            result = inspect_image(path)
            self.assertFalse(result["valid"])
            self.assertTrue(result["error"])

    def test_decoded_duplicates_across_splits_and_originals_preserved(self):
        image = Image.new("RGB", (12, 8), color=(20, 40, 60))
        first = self.root / "first.png"
        second = self.root / "second.png"
        image.save(first, compress_level=0)
        image.save(second, compress_level=9)
        before = {path: path.read_bytes() for path in (first, second)}
        records = [inspect_image(path) for path in (first, second)]
        self.assertNotEqual(records[0]["file_sha256"], records[1]["file_sha256"])
        self.assertEqual(records[0]["pixel_sha256"], records[1]["pixel_sha256"])
        manifest = pd.DataFrame(records)
        manifest["image_id"] = ["Cat_1", "Cat_2"]
        manifest["official_split"] = ["trainval", "test"]
        duplicates = find_exact_duplicates(manifest)
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(duplicates.cross_official_splits.all())
        for path in before:
            self.assertEqual(path.read_bytes(), before[path])

    def test_no_duplicates_has_usable_empty_report(self):
        path = self.root / "single.png"
        Image.new("L", (10, 8), 100).save(path)
        result = inspect_image(path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["original_mode"], "L")
        manifest = pd.DataFrame([{**result, "image_id": "Cat_1", "official_split": "trainval"}])
        duplicates = find_exact_duplicates(manifest)
        self.assertTrue(duplicates.empty)
        self.assertIn("cross_official_splits", duplicates.columns)


if __name__ == "__main__":
    unittest.main()
