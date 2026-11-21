"""Contrôle le prétraitement sans charger de modèle ni de données externes."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch

from ianimaux.datasets import FitInsideSquare, PetImageDataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # A fourth channel exercises conversion of the RGBA images found in the audit.
        Image.new("RGBA", (300, 250), (150, 124, 96, 255)).save(self.root / "animal.png")

    def dataset(self, split, path="animal.png"):
        csv_path = self.root / f"{split}.csv"
        pd.DataFrame([{"relative_path": path, "target": 0, "split": split}]).to_csv(csv_path, index=False)
        return PetImageDataset(csv_path, self.root)

    def test_validation_is_rgb_normalized_and_repeatable(self):
        dataset = self.dataset("validation")
        first, label = dataset[0]
        second, _ = dataset[0]
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(label, 0)
        self.assertTrue(torch.equal(first, second))
        expected_red = (150 / 255 - 0.485) / 0.229
        self.assertAlmostEqual(first[0, 112, 112].item(), expected_red, places=5)

    def test_augmentations_are_only_attached_to_training(self):
        patterned = Image.new("RGB", (300, 250))
        patterned.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(250) for x in range(300)])
        patterned.save(self.root / "animal.png")
        train = self.dataset("train")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            samples = [train[0][0] for _ in range(3)]
        self.assertTrue(any(not torch.equal(samples[0], other) for other in samples[1:]))
        for split in ("validation", "test"):
            tensor, _ = self.dataset(split)[0]
            self.assertTrue(torch.equal(tensor, self.dataset(split)[0][0]))

    def test_relative_path_cannot_escape_data_directory(self):
        with self.assertRaisesRegex(ValueError, "sort du dossier"):
            self.dataset("validation", "../outside.png")

    def test_letterbox_keeps_top_and_bottom_of_portrait_without_distortion(self):
        portrait = Image.new("RGB", (100, 200), "white")
        portrait.paste((255, 0, 0), (0, 0, 100, 30))
        portrait.paste((0, 0, 255), (0, 170, 100, 200))
        result = FitInsideSquare()(portrait)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((112, 0)), (255, 0, 0))
        self.assertEqual(result.getpixel((112, 223)), (0, 0, 255))
        self.assertEqual(result.getpixel((0, 112)), (124, 116, 104))


if __name__ == "__main__":
    unittest.main()
