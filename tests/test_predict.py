"""Contrôles d'inférence, de lecture et d'isolation des erreurs de photos."""

from io import BytesIO
import unittest

from PIL import Image
import torch
from torch import nn

from ianimaux.predict import classify_uploads, InvalidPhotoError, read_photo, upload_signature


def photo_file(name="photo.png", color=(150, 124, 96), mode="RGB"):
    file = BytesIO()
    Image.new(mode, (40, 30), color).save(file, format="PNG")
    file.name = name
    return file


class FixedModel(nn.Module):
    def forward(self, images):
        return torch.tensor([[0., 2.]]).expand(len(images), -1)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_invalid_photo_does_not_interrupt_other_images(self):
        invalid = BytesIO(b"not an image")
        invalid.name = "corrupt.jpg"
        first = photo_file()
        second = photo_file("other.png", (10, 20, 30))
        progress = []
        result = classify_uploads(FixedModel(), [first, invalid, second], lambda current, total: progress.append((current, total)))
        self.assertEqual([row["category"] for row in result], ["Chien", None, "Chien"])
        self.assertTrue(result[1]["error"])
        self.assertIsNone(result[1]["confidence"])
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertAlmostEqual(result[0]["confidence"], torch.tensor([0., 2.]).softmax(0)[1].item())
        self.assertAlmostEqual(result[0]["p_chat"] + result[0]["p_chien"], 1, places=6)

    def test_original_bytes_and_file_position_are_preserved(self):
        file = photo_file(mode="RGBA", color=(150, 124, 96, 255))
        original = file.getvalue()
        file.seek(5)
        result = classify_uploads(FixedModel(), [file])
        self.assertEqual(file.getvalue(), original)
        self.assertEqual(file.tell(), 5)
        self.assertTrue(result[0]["preview"])
        with Image.open(BytesIO(result[0]["preview"])) as preview:
            self.assertEqual(preview.mode, "RGB")

    def test_signature_detects_new_content_under_same_name(self):
        first = photo_file("same.png")
        other = photo_file("same.png", (1, 2, 3))
        self.assertNotEqual(upload_signature([first]), upload_signature([other]))
        result = classify_uploads(FixedModel(), [first, other])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["filename"], result[1]["filename"])

    def test_exif_orientation_is_applied(self):
        buffer = BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20)).save(buffer, format="JPEG", exif=exif)
        self.assertEqual(read_photo(buffer.getvalue()).size, (20, 40))

    def test_empty_and_unsupported_contents_are_reported(self):
        with self.assertRaisesRegex(InvalidPhotoError, "vide"):
            read_photo(b"")
        buffer = BytesIO()
        Image.new("RGB", (20, 20)).save(buffer, format="GIF")
        with self.assertRaisesRegex(InvalidPhotoError, "Format"):
            read_photo(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
