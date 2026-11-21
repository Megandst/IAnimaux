"""Vérifie les garanties de séparation, pas une performance de classification."""

import hashlib
import unittest

import pandas as pd

from ianimaux.data import create_partitions


def sample_manifest():
    rows = []
    for class_id, species in ((1, "Chat"), (2, "Chien")):
        for index in range(14):
            image_id = f"breed{class_id}_{index}"
            rows.append({
                "image_id": image_id, "relative_path": f"images/{image_id}.jpg",
                "official_split": "trainval" if index < 10 else "test",
                "class_id": class_id, "breed": f"breed{class_id}",
                "species_id": class_id, "species": species, "target": class_id - 1,
                "valid": True,
                "file_sha256": hashlib.sha256(f"file{image_id}".encode()).hexdigest(),
                "pixel_sha256": hashlib.sha256(f"pixels{image_id}".encode()).hexdigest(),
            })
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_stratification_coverage_and_official_test_preservation(self):
        source = sample_manifest()
        partitions = create_partitions(source)
        self.assertEqual({name: len(frame) for name, frame in partitions.items()}, {"train": 16, "validation": 4, "test": 8})
        self.assertEqual(partitions["validation"].species.value_counts().to_dict(), {"Chat": 2, "Chien": 2})
        combined = pd.concat(partitions.values())
        self.assertEqual(set(combined.image_id), set(source.image_id))
        self.assertFalse(combined.image_id.duplicated().any())
        expected_test = source.loc[source.official_split.eq("test")].sort_values("image_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(partitions["test"].drop(columns="split"), expected_test)

    def test_same_seed_is_independent_of_manifest_order(self):
        source = sample_manifest()
        first = create_partitions(source, seed=42)
        second = create_partitions(source.sample(frac=1, random_state=7), seed=42)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_other_seed_changes_trainval_but_never_test(self):
        first = create_partitions(sample_manifest(), seed=42)
        second = create_partitions(sample_manifest(), seed=7)
        self.assertNotEqual(set(first["train"].image_id), set(second["train"].image_id))
        pd.testing.assert_frame_equal(first["test"], second["test"])

    def test_invalid_image_blocks_split(self):
        source = sample_manifest()
        source.loc[0, "valid"] = False
        with self.assertRaisesRegex(ValueError, "invalides"):
            create_partitions(source)

    def test_identical_pixels_under_different_names_block_split(self):
        source = sample_manifest()
        source.loc[13, "pixel_sha256"] = source.loc[0, "pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "Doublons"):
            create_partitions(source)

    def test_inverted_binary_target_is_rejected(self):
        source = sample_manifest()
        source.loc[0, "target"] = 1
        with self.assertRaisesRegex(ValueError, "cible"):
            create_partitions(source)


if __name__ == "__main__":
    unittest.main()
