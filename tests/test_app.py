"""Parcours Streamlit avec importations réelles et checkpoint local entraîné."""

from io import BytesIO
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
from streamlit.testing.v1 import AppTest
import streamlit as st

from config import MODEL_PATH, PROJECT_ROOT


def png_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="PNG")
    return buffer.getvalue()


class AppTests(unittest.TestCase):
    def test_empty_selection_disables_classification(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.file_uploader[0].accept_multiple_files)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.download_button), 0)

    @unittest.skipUnless(MODEL_PATH.is_file(), "Le checkpoint entraîné local est nécessaire au parcours complet.")
    def test_upload_classify_keep_results_and_reset_when_selection_changes(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        app.file_uploader[0].set_value([
            ("photo.png", png_bytes((150, 124, 96)), "image/png"),
            ("corrupt.jpg", b"invalid", "image/jpeg"),
        ]).run(timeout=30)
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertEqual(app.dataframe[0].value.Statut.tolist(), ["Classée", "Erreur"])
        self.assertEqual(app.metric[0].value, "1")
        self.assertEqual(app.metric[3].value, "1")
        app.run(timeout=30)
        self.assertEqual(len(app.dataframe[0].value), 2)
        app.file_uploader[0].set_value(("photo.png", png_bytes((1, 2, 3)), "image/png")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)
        app.file_uploader[0].clear().run(timeout=30)
        self.assertTrue(app.button[0].disabled)

    def test_missing_checkpoint_is_reported_without_crash(self):
        with tempfile.TemporaryDirectory() as directory, patch("config.MODEL_PATH", Path(directory) / "missing.pt"):
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)
        self.assertIn("introuvable", app.error[0].value)

    @unittest.skipUnless(MODEL_PATH.is_file(), "Le checkpoint local est nécessaire.")
    def test_annotation_persists_but_does_not_override_future_predictions(self):
        model_before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory, patch("config.FEEDBACK_DB", Path(directory) / "corrections.sqlite3"):
            photo = ("photo.png", png_bytes((150, 124, 96)), "image/png")
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
            app.file_uploader[0].set_value(photo).run(timeout=30)
            app.button[0].click().run(timeout=30)
            original = app.dataframe[0].value.iloc[0]
            corrected = "Chat" if original["Prédiction IA"] == "Chien" else "Chien"
            app.selectbox[1].set_value(corrected)
            app.text_input[0].set_value("Correction de test")
            app.button[1].click().run(timeout=30)
            self.assertFalse(app.exception)
            row = app.dataframe[0].value.iloc[0]
            self.assertEqual(row["Catégorie retenue"], corrected)
            self.assertEqual(row["Origine"], "Correction humaine du lot")
            self.assertEqual(row["Prédiction IA"], original["Prédiction IA"])
            self.assertEqual(row["Confiance IA"], original["Confiance IA"])

            fresh = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
            fresh.file_uploader[0].set_value(photo).run(timeout=30)
            fresh.button[0].click().run(timeout=30)
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.dataframe[0].value.iloc[0]["Catégorie retenue"], original["Prédiction IA"])
            self.assertEqual(fresh.dataframe[0].value.iloc[0]["Origine"], "Modèle")
            self.assertEqual(fresh.selectbox[1].value, corrected)
            self.assertTrue(any("Annotation déjà enregistrée" in item.value for item in fresh.caption))
        self.assertEqual(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), model_before)

    @unittest.skipUnless(MODEL_PATH.is_file(), "Le checkpoint local est nécessaire.")
    def test_download_contains_original_and_follows_updated_batch_correction(self):
        with tempfile.TemporaryDirectory() as directory, patch("config.FEEDBACK_DB", Path(directory) / "corrections.sqlite3"), patch("streamlit.download_button", wraps=st.download_button) as download:
            content = png_bytes((150, 124, 96))
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
            app.file_uploader[0].set_value([("photo.png", content, "image/png"), ("bad.jpg", b"invalid", "image/jpeg")]).run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.download_button[0].disabled)
            first_callback = download.call_args.kwargs["data"]
            predicted = app.dataframe[0].value.iloc[0]["Prédiction IA"]
            corrected = "Chat" if predicted == "Chien" else "Chien"
            with ZipFile(BytesIO(first_callback())) as archive:
                self.assertEqual(set(archive.namelist()), {"chat/", "chien/", f"{predicted.lower()}/photo.png"})
                self.assertEqual(archive.read(f"{predicted.lower()}/photo.png"), content)
            app.selectbox[1].set_value(corrected)
            app.button[1].click().run(timeout=30)
            self.assertFalse(app.exception)
            with ZipFile(BytesIO(download.call_args.kwargs["data"]())) as archive:
                self.assertIn(f"{corrected.lower()}/photo.png", archive.namelist())
                self.assertNotIn(f"{predicted.lower()}/photo.png", archive.namelist())
            app.button[0].click().run(timeout=30)
            with ZipFile(BytesIO(download.call_args.kwargs["data"]())) as archive:
                self.assertIn(f"{predicted.lower()}/photo.png", archive.namelist())

    @unittest.skipUnless(MODEL_PATH.is_file(), "Le checkpoint local est nécessaire.")
    def test_download_is_disabled_when_all_files_are_invalid(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        app.file_uploader[0].set_value(("bad.jpg", b"invalid", "image/jpeg")).run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.download_button[0].disabled)


if __name__ == "__main__":
    unittest.main()
