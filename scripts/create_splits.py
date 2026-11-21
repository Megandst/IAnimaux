"""Fige les partitions reproductibles à partir du manifeste audité."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_ROOT, EXPLORATION_DIR, SEED, SPLITS_DIR, VALIDATION_FRACTION
from ianimaux.data import create_partitions, read_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EXPLORATION_DIR / "manifest.csv")
    parser.add_argument("--data-dir", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    annotations = read_annotations(args.data_dir)
    # Compare against the official files rather than trusting a manually edited CSV.
    pd.testing.assert_frame_equal(
        manifest[list(annotations.columns)].sort_values("image_id").reset_index(drop=True),
        annotations.sort_values("image_id").reset_index(drop=True),
    )
    partitions = create_partitions(manifest, args.seed, args.validation_fraction)
    print("Vérification des empreintes des fichiers actuels…", flush=True)
    root = args.data_dir.resolve()
    for row in manifest.itertuples():
        path = (root / row.relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Chemin hors du dossier des données : {row.relative_path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError(f"Image modifiée depuis l'audit : {row.image_id}. Refaire l'exploration.")

    # Refuse to silently replace an existing scientific split with another one.
    csv_contents = {name: frame.to_csv(index=False) for name, frame in partitions.items()}
    for name, content in csv_contents.items():
        path = args.output_dir / f"{name}.csv"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"{path} existe avec une autre séparation. Utiliser un autre --output-dir.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in csv_contents.items():
        (args.output_dir / f"{name}.csv").write_text(content)
    combined = pd.concat(partitions.values(), ignore_index=True)
    combined.groupby(["split", "species"]).size().rename("count").to_csv(args.output_dir / "effectifs_especes.csv")
    combined.groupby(["split", "class_id", "breed", "species"]).size().rename("count").to_csv(args.output_dir / "effectifs_races.csv")
    majority = partitions["train"].target.value_counts().idxmax()
    summary = {
        "seed": args.seed,
        "validation_fraction_of_trainval": args.validation_fraction,
        "stratify_by": "class_id (race)",
        "counts": {name: len(frame) for name, frame in partitions.items()},
        "species_counts": {name: {key: int(value) for key, value in frame.species.value_counts().items()} for name, frame in partitions.items()},
        "breeds_per_split": {name: int(frame.class_id.nunique()) for name, frame in partitions.items()},
        "official_test_members_preserved": set(partitions["test"].image_id) == set(annotations.loc[annotations.official_split.eq("test"), "image_id"]),
        "overlapping_image_ids": int(combined.image_id.duplicated().sum()),
        "overlapping_pixel_hashes": int(combined.pixel_sha256.duplicated().sum()),
        "all_current_file_hashes_verified": True,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "partition_sha256": {name: hashlib.sha256((args.output_dir / f"{name}.csv").read_bytes()).hexdigest() for name in partitions},
        "sklearn_version": metadata.version("scikit-learn"),
        "pandas_version": metadata.version("pandas"),
        "label_mapping": {"0": "Chat", "1": "Chien"},
        "training_majority_target": int(majority),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
