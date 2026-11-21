"""Corrections humaines persistantes ; aucune modification des poids du modèle."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3

import pandas as pd

from config import FEEDBACK_DB, SPLITS_DIR
from ianimaux.model import CLASS_NAMES
from ianimaux.predict import read_photo


def existing_source(file_hash, pixel_hash, splits_dir):
    """Tout exemple Oxford connu est exclu des nouveaux exemples d'entraînement."""
    frames = {}
    for split in ("train", "validation", "test"):
        path = Path(splits_dir) / f"{split}.csv"
        if not path.is_file():
            return "unverified"
        frames[split] = pd.read_csv(path, usecols=["file_sha256", "pixel_sha256"])
    for split in ("test", "validation", "train"):
        frame = frames[split]
        if (frame.file_sha256.eq(file_hash) | frame.pixel_sha256.eq(pixel_hash)).any():
            return split
    return "external"


def get_corrections(image_hashes, db_path=FEEDBACK_DB):
    wanted = set(image_hashes)
    if not wanted or not Path(db_path).is_file():
        return {}
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT image_sha256, corrected_category, note, source_split, training_eligible, updated_at FROM corrections"
        )
        return {row["image_sha256"]: dict(row) for row in rows if row["image_sha256"] in wanted}


def save_correction(content, filename, category, prediction, model_sha256,
                    note="", db_path=FEEDBACK_DB, splits_dir=SPLITS_DIR):
    if category not in CLASS_NAMES or prediction.get("category") not in CLASS_NAMES:
        raise ValueError("Une catégorie Chat ou Chien est requise.")
    confidence = float(prediction["confidence"])
    if not 0 <= confidence <= 1 or not re.fullmatch(r"[a-f0-9]{64}", model_sha256):
        raise ValueError("Métadonnées du modèle invalides.")
    if len(note) > 500:
        raise ValueError("Le commentaire est limité à 500 caractères.")
    image = read_photo(content)
    image_hash = hashlib.sha256(content).hexdigest()
    if prediction.get("image_sha256") != image_hash:
        raise ValueError("La correction ne correspond pas à l'image analysée.")
    pixel_hash = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
    source = existing_source(image_hash, pixel_hash, splits_dir)
    now = datetime.now(timezone.utc).isoformat()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                image_sha256 TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                original_filename TEXT NOT NULL,
                pixel_sha256 TEXT NOT NULL,
                corrected_category TEXT NOT NULL CHECK(corrected_category IN ('Chat', 'Chien')),
                model_category TEXT NOT NULL,
                model_confidence REAL NOT NULL,
                model_sha256 TEXT NOT NULL,
                note TEXT NOT NULL,
                source_split TEXT NOT NULL,
                training_eligible INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS correction_events (
                id INTEGER PRIMARY KEY,
                image_sha256 TEXT NOT NULL,
                corrected_category TEXT NOT NULL,
                model_category TEXT NOT NULL,
                model_confidence REAL NOT NULL,
                model_sha256 TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_sha256) DO UPDATE SET
                corrected_category=excluded.corrected_category,
                model_category=excluded.model_category,
                model_confidence=excluded.model_confidence,
                model_sha256=excluded.model_sha256,
                note=excluded.note,
                source_split=excluded.source_split,
                training_eligible=excluded.training_eligible,
                updated_at=excluded.updated_at
        """, (image_hash, content, filename, pixel_hash, category, prediction["category"], confidence,
              model_sha256, note.strip(), source, int(source == "external"), now, now))
        connection.execute("""
            INSERT INTO correction_events
                (image_sha256, corrected_category, model_category, model_confidence, model_sha256, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (image_hash, category, prediction["category"], confidence, model_sha256, note.strip(), now))
    return get_corrections([image_hash], db_path)[image_hash]


def validate_training_export(folder, splits_dir=SPLITS_DIR):
    """Vérifie chaque nouvel exemple avant de le joindre à l'entraînement."""
    root = Path(folder).resolve()
    frame = pd.read_csv(root / "train.csv")
    required = {"relative_path", "image_sha256", "target", "species", "split", "source"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Export de corrections vide ou incomplet.")
    if frame.image_sha256.duplicated().any() or frame.relative_path.duplicated().any():
        raise ValueError("Doublons dans l'export de corrections.")
    if not frame.split.eq("train").all() or not frame.source.eq("human_feedback").all():
        raise ValueError("Seuls des exemples humains d'entraînement sont autorisés.")
    for row in frame.itertuples():
        path = (root / row.relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Chemin de correction hors du dossier d'export.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row.image_sha256:
            raise ValueError("Photo corrigée modifiée depuis l'export.")
        if row.target not in (0, 1) or row.species != CLASS_NAMES[row.target]:
            raise ValueError("Étiquette humaine incohérente.")
        image = read_photo(content)
        pixel_hash = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
        if existing_source(row.image_sha256, pixel_hash, splits_dir) != "external":
            raise ValueError("Une correction recoupe un ensemble Oxford connu ou non vérifiable.")
    return frame
