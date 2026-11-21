"""Les corrections sont des labels humains, distincts d'un apprentissage du réseau."""

from contextlib import closing
import hashlib
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd
from PIL import Image

from ianimaux.feedback import get_corrections, save_correction, validate_training_export
from ianimaux.predict import InvalidPhotoError
from scripts.export_feedback import export_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "feedback" / "corrections.sqlite3"
        self.splits = self.root / "splits"
        self.splits.mkdir()
        for split in ("train", "validation", "test"):
            pd.DataFrame(columns=["file_sha256", "pixel_sha256"]).to_csv(self.splits / f"{split}.csv", index=False)
        self.image = Image.new("RGB", (40, 30), (150, 124, 96))
        stream = BytesIO()
        self.image.save(stream, format="PNG", compress_level=0)
        self.content = stream.getvalue()
        self.image_hash = hashlib.sha256(self.content).hexdigest()
        self.model_hash = hashlib.sha256(b"model").hexdigest()
        self.prediction = {"category": "Chien", "confidence": 0.541, "image_sha256": self.image_hash}

    def save(self, category="Chat", content=None, prediction=None):
        return save_correction(
            self.content if content is None else content, "../../same.png", category,
            self.prediction if prediction is None else prediction, self.model_hash,
            note="Sphynx", db_path=self.db, splits_dir=self.splits,
        )

    def test_correction_survives_reopening_and_does_not_replace_model_prediction(self):
        record = self.save()
        self.assertTrue(record["training_eligible"])
        restored = get_corrections([self.image_hash], self.db)[self.image_hash]
        self.assertEqual(restored["corrected_category"], "Chat")
        with closing(sqlite3.connect(self.db)) as connection:
            stored = connection.execute("SELECT content, model_category, model_confidence FROM corrections").fetchone()
        self.assertEqual(stored, (self.content, "Chien", 0.541))

    def test_repeated_correction_updates_one_photo_and_keeps_event_history(self):
        self.save()
        self.save("Chien")
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM correction_events").fetchone()[0], 2)
        self.assertEqual(get_corrections([self.image_hash], self.db)[self.image_hash]["corrected_category"], "Chien")

    def test_holdout_photo_is_memorized_but_excluded_from_export_even_after_reencoding(self):
        pixel_hash = hashlib.sha256(b"40x30:RGB:" + self.image.tobytes()).hexdigest()
        # Same decoded pixels, different file fingerprint: it must still be protected.
        pd.DataFrame([{"file_sha256": "different", "pixel_sha256": pixel_hash}]).to_csv(self.splits / "validation.csv", index=False)
        record = self.save()
        self.assertEqual(record["source_split"], "validation")
        self.assertFalse(record["training_eligible"])
        summary = export_feedback(self.db, self.root / "export", self.splits)
        self.assertEqual(summary["exported_examples"], 0)
        self.assertEqual(summary["excluded"][0]["reason"], "validation")

    def test_missing_partitions_never_allow_unverified_training_examples(self):
        record = save_correction(self.content, "photo.png", "Chat", self.prediction, self.model_hash, db_path=self.db, splits_dir=self.root / "missing")
        self.assertEqual(record["source_split"], "unverified")
        self.assertFalse(record["training_eligible"])

    def test_export_uses_hash_paths_preserves_bytes_and_refuses_overwrite(self):
        self.save()
        folder = self.root / "export"
        summary = export_feedback(self.db, folder, self.splits)
        self.assertEqual(summary["exported_examples"], 1)
        frame = pd.read_csv(folder / "train.csv")
        self.assertEqual(frame.target.tolist(), [0])
        self.assertEqual((folder / frame.relative_path.iloc[0]).read_bytes(), self.content)
        self.assertTrue((folder / frame.relative_path.iloc[0]).resolve().is_relative_to(folder.resolve()))
        with self.assertRaisesRegex(ValueError, "existe"):
            export_feedback(self.db, folder, self.splits)

    def test_invalid_label_and_wrong_image_cannot_be_saved(self):
        with self.assertRaises(ValueError):
            self.save("Sphynx")
        other = {**self.prediction, "image_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "ne correspond pas"):
            self.save(prediction=other)
        with self.assertRaises(InvalidPhotoError):
            self.save(content=b"invalid")
        self.assertFalse(self.db.exists())

    def test_training_rechecks_holdout_membership_after_export(self):
        self.save()
        folder = self.root / "export"
        export_feedback(self.db, folder, self.splits)
        self.assertEqual(len(validate_training_export(folder, self.splits)), 1)
        pd.DataFrame([{"file_sha256": self.image_hash, "pixel_sha256": "other"}]).to_csv(self.splits / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Oxford"):
            validate_training_export(folder, self.splits)


if __name__ == "__main__":
    unittest.main()
