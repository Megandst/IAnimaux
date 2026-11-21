"""Exporte les corrections externes pour préparer un futur entraînement supervisé."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEEDBACK_DB, PROJECT_ROOT, SPLITS_DIR
from ianimaux.feedback import existing_source
from ianimaux.model import CLASS_NAMES


def export_feedback(db_path, output_dir, splits_dir=SPLITS_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Ce dossier d'export existe déjà ; choisir un nouveau dossier pour préserver l'instantané.")
    if not Path(db_path).is_file():
        raise ValueError("Aucune correction enregistrée.")
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        records = [dict(row) for row in connection.execute("SELECT * FROM corrections ORDER BY image_sha256")]
    eligible, excluded = [], []
    for row in records:
        if hashlib.sha256(row["content"]).hexdigest() != row["image_sha256"]:
            raise ValueError("L'intégrité d'une photo corrigée n'est pas vérifiée.")
        # Recheck current partitions: the stored eligibility flag may be outdated.
        source = existing_source(row["image_sha256"], row["pixel_sha256"], splits_dir)
        if source == "external":
            eligible.append(row)
        else:
            excluded.append({"image_sha256": row["image_sha256"], "reason": source})
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    manifest = []
    for row in eligible:
        with Image.open(BytesIO(row["content"])) as image:
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image.format]
        relative = f"images/{row['image_sha256']}{suffix}"
        (output_dir / relative).write_bytes(row["content"])
        manifest.append({
            "image_id": f"feedback_{row['image_sha256']}", "relative_path": relative,
            "split": "train", "target": CLASS_NAMES.index(row["corrected_category"]),
            "species": row["corrected_category"], "source": "human_feedback",
            "original_filename": row["original_filename"], "note": row["note"],
            "image_sha256": row["image_sha256"],
        })
    columns = ["image_id", "relative_path", "split", "target", "species", "source", "original_filename", "note", "image_sha256"]
    pd.DataFrame(manifest, columns=columns).to_csv(output_dir / "train.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recorded_corrections": len(records), "exported_examples": len(eligible),
        "excluded": excluded, "model_retrained": False,
        "note": "Instantané d'exemples humains à examiner ; non consommé automatiquement par scripts/train.py.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=FEEDBACK_DB)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "feedback" / "export_01")
    args = parser.parse_args()
    print(json.dumps(export_feedback(args.db, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
