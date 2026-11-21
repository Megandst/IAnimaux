"""Vérifie les lots d'images d'entraînement et de validation, sans entraîner."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_ROOT, SEED, SPLITS_DIR
from ianimaux.datasets import PetImageDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("La taille de lot doit être positive.")
    torch.manual_seed(SEED)
    summary = {}
    for name in ("train", "validation"):
        dataset = PetImageDataset(args.splits_dir / f"{name}.csv", args.data_dir)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            num_workers=0, generator=torch.Generator().manual_seed(SEED),
        )
        images, targets = next(iter(loader))
        expected = (min(args.batch_size, len(dataset)), 3, 224, 224)
        if tuple(images.shape) != expected or images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError(f"Lot d'images incorrect pour {name}.")
        if targets.dtype != torch.int64 or not torch.isin(targets, torch.tensor([0, 1])).all():
            raise ValueError(f"Cibles incorrectes pour {name}.")
        summary[name] = {"dataset_size": len(dataset), "batch_shape": list(images.shape), "targets": targets.tolist()}
        if name == "validation":
            first, label = dataset[0]
            second, repeated_label = dataset[0]
            if not torch.equal(first, second) or label != repeated_label:
                raise ValueError("La validation doit être déterministe.")
            summary[name]["repeated_image_identical"] = True
    # Construct the test dataset without reading or visualizing any test photograph.
    test = PetImageDataset(args.splits_dir / "test.csv", args.data_dir)
    summary["test"] = {"dataset_size": len(test), "images_loaded": 0}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.splits_dir / "loader_checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("Lots vérifiés. Aucun modèle construit ou entraîné.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
