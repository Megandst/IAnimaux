"""Premier entraînement CPU du classificateur binaire, sans utiliser le test."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from config import DATASET_ROOT, SEED, SPLITS_DIR
from ianimaux.datasets import PetImageDataset
from ianimaux.feedback import validate_training_export
from ianimaux.model import build_model, CLASS_NAMES, load_checkpoint, preprocessing_metadata
from ianimaux.training import run_epoch


def save_figures(history, predictions, output_dir):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, label in (("train", "Entraînement"), ("validation", "Validation")):
        axes[0].plot(frame.epoch, frame[f"{name}_loss"], marker="o", label=label)
        axes[1].plot(frame.epoch, frame[f"{name}_accuracy"], marker="o", label=label)
    axes[0].set(xlabel="Époque", ylabel="Entropie croisée", title="Évolution de la perte")
    axes[1].set(xlabel="Époque", ylabel="Exactitude", title="Évolution de l'exactitude", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.set_xticks(frame.epoch)
    figure.tight_layout()
    figure.savefig(output_dir / "courbes_apprentissage.png", dpi=150)
    plt.close(figure)
    matrix = confusion_matrix(predictions.target, predictions.prediction, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set(xlabel="Classe prédite", ylabel="Classe réelle", title="Validation — modèle sélectionné")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "matrice_confusion_validation.png", dpi=150)
    plt.close(display.figure_)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Chronométrer 3 lots d'entraînement et 2 de validation, sans sauvegarder de modèle.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default="first_run")
    parser.add_argument("--preprocessing", choices=["center_crop", "letterbox"], default="center_crop")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialiser depuis un checkpoint local plutôt qu'ImageNet.")
    parser.add_argument("--feedback-dir", type=Path, help="Instantané de corrections externes revues, contenant train.csv.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.learning_rate <= 0:
        parser.error("Époques, taille de lot, threads et taux d'apprentissage doivent être positifs.")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("Le nom d'expérience doit être un nom simple, sans chemin.")

    result_root = PROJECT_ROOT / "results" / "training"
    output_dir = result_root / args.run_name
    model_dir = PROJECT_ROOT / "models" / args.run_name
    if not args.benchmark and (output_dir.exists() or model_dir.exists()):
        parser.error("Cette expérience existe déjà. Choisir un autre --run-name pour préserver ses résultats.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(PROJECT_ROOT / ".cache" / "torch"))
    split_summary = json.loads((SPLITS_DIR / "summary.json").read_text())
    feedback_frame = None
    if args.feedback_dir:
        feedback_frame = validate_training_export(args.feedback_dir)
    datasets, loaders = {}, {}
    training_targets = None
    for name in ("train", "validation"):
        csv_path = SPLITS_DIR / f"{name}.csv"
        current_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if current_hash != split_summary["partition_sha256"][name]:
            raise ValueError(f"La partition {name} diffère de sa référence enregistrée.")
        datasets[name] = PetImageDataset(csv_path, DATASET_ROOT, preprocessing=args.preprocessing)
        if datasets[name].split != name:
            raise ValueError(f"Rôle de partition incorrect : {name}")
        if name == "train":
            training_targets = datasets[name].frame.target
            if feedback_frame is not None:
                feedback = PetImageDataset(args.feedback_dir / "train.csv", args.feedback_dir, preprocessing=args.preprocessing)
                training_targets = pd.concat([training_targets, feedback.frame.target], ignore_index=True)
                datasets[name] = ConcatDataset([datasets[name], feedback])
        loaders[name] = DataLoader(
            datasets[name], batch_size=args.batch_size, shuffle=name == "train",
            num_workers=0, generator=torch.Generator().manual_seed(args.seed),
        )

    if args.init_checkpoint:
        print(f"Initialisation depuis le checkpoint local : {args.init_checkpoint}", flush=True)
        model, _ = load_checkpoint(args.init_checkpoint)
    else:
        print("Chargement MobileNetV3 Small ImageNet (environ 10 Mo si absent du cache)…", flush=True)
        model = build_model(pretrained=True)
    model.preprocessing_name = args.preprocessing
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    settings = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "optimizer": "Adam",
        "loss": "CrossEntropyLoss", "seed": args.seed, "cpu_threads": args.threads,
        "device": "cpu", "num_workers": 0,
        "weights_source": "local_checkpoint" if args.init_checkpoint else "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "initial_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest() if args.init_checkpoint else None,
        "preprocessing": args.preprocessing,
        "feedback_examples": 0 if feedback_frame is None else len(feedback_frame),
        "training_images": len(datasets["train"]),
        "feedback_manifest_sha256": hashlib.sha256((args.feedback_dir / "train.csv").read_bytes()).hexdigest() if args.feedback_dir else None,
        "frozen_features_and_batchnorm": True,
        "selection_criterion": "minimum validation loss",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "partition_sha256": {name: split_summary["partition_sha256"][name] for name in datasets},
        "torch_version": str(torch.__version__), "torchvision_version": metadata.version("torchvision"),
        "training_augmentation": {"resize_short_side": 256, "random_crop": 224, "horizontal_flip_probability": 0.5},
    }
    if args.preprocessing == "letterbox":
        settings["training_augmentation"] = {"fit_inside_square": 224, "horizontal_flip_probability": 0.5, "random_crop": False}
    print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)

    if args.benchmark:
        train_metrics, _ = run_epoch(model, loaders["train"], criterion, optimizer, max_batches=3)
        validation_metrics, _ = run_epoch(model, loaders["validation"], criterion, max_batches=2)
        estimate = (
            train_metrics["seconds"] / train_metrics["images"] * len(datasets["train"])
            + validation_metrics["seconds"] / validation_metrics["images"] * len(datasets["validation"])
        )
        result = {
            "settings": settings, "train": train_metrics, "validation": validation_metrics,
            "estimated_seconds_per_epoch": estimate, "estimated_total_seconds": estimate * args.epochs,
            "purpose": "Contrôle de durée sur quelques lots ; les scores ne mesurent pas la performance du modèle.",
            "checkpoint_saved": False,
        }
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / f"benchmark_{args.run_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Estimation : {estimate:.1f} s par époque, {estimate * args.epochs:.1f} s pour {args.epochs} époques (indicatif).", flush=True)
        return 0

    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    if args.init_checkpoint:
        initial_metrics, initial_rows = run_epoch(model, loaders["validation"], criterion)
        initial_predictions = pd.DataFrame(initial_rows)
        initial_predictions.insert(0, "image_id", datasets["validation"].frame.image_id.tolist())
        initial_predictions.to_csv(output_dir / "initial_validation_predictions.csv", index=False)
        (output_dir / "initial_validation_metrics.json").write_text(json.dumps(initial_metrics, indent=2) + "\n")
        print(f"Contrôle avant réentraînement avec le nouveau prétraitement : perte={initial_metrics['loss']:.4f}, exactitude={initial_metrics['accuracy']:.2%}", flush=True)
    history, best_loss = [], float("inf")
    best_epoch, best_predictions, best_validation = None, None, None
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"Époque {epoch}/{args.epochs}…", flush=True)
        train_metrics, _ = run_epoch(model, loaders["train"], criterion, optimizer)
        validation_metrics, rows = run_epoch(model, loaders["validation"], criterion)
        entry = {"epoch": epoch}
        for name, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            entry.update({f"{name}_{key}": value for key, value in metrics.items()})
        history.append(entry)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"Perte train={train_metrics['loss']:.4f}, validation={validation_metrics['loss']:.4f} ; exactitude validation={validation_metrics['accuracy']:.2%}", flush=True)
        if validation_metrics["loss"] < best_loss:
            best_loss, best_epoch, best_validation = validation_metrics["loss"], epoch, validation_metrics
            best_predictions = pd.DataFrame(rows)
            best_predictions.insert(0, "image_id", datasets["validation"].frame.image_id.tolist())
            best_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
            torch.save({
                "architecture": "mobilenet_v3_small", "class_names": CLASS_NAMES,
                "preprocessing": preprocessing_metadata(args.preprocessing), "model_state_dict": model.state_dict(),
                "epoch": epoch, "validation_metrics": validation_metrics, "settings": settings,
            }, model_dir / "best_model.pt")

    elapsed = perf_counter() - start
    report = classification_report(best_predictions.target, best_predictions.prediction, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    majority = int(training_targets.value_counts().idxmax())
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch, "selection_criterion": "minimum validation loss",
        "validation_metrics": best_validation, "classification_report": report,
        "confusion_matrix": confusion_matrix(best_predictions.target, best_predictions.prediction, labels=[0, 1]).tolist(),
        "majority_baseline": {"target_from_train": majority, "class": CLASS_NAMES[majority], "validation_accuracy": float(datasets["validation"].frame.target.eq(majority).mean())},
        "training_and_validation_seconds": elapsed, "test_used": False,
        "checkpoint_relative_path": (model_dir / "best_model.pt").relative_to(PROJECT_ROOT).as_posix(),
    }
    # Verify the saved model, rather than assuming serialization succeeded.
    restored, checkpoint = load_checkpoint(model_dir / "best_model.pt")
    examples = torch.stack([datasets["validation"][index][0] for index in range(min(2, len(datasets["validation"])))])
    with torch.inference_mode():
        probabilities = restored(examples).softmax(dim=1)
    expected = torch.tensor(best_predictions[["p_chat", "p_chien"]].head(len(examples)).values, dtype=torch.float32)
    if not torch.allclose(probabilities, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("Les probabilités du modèle rechargé diffèrent des prédictions sauvegardées.")
    summary["checkpoint_reload_verified"] = True
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    save_figures(history, best_predictions, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Résultats : {output_dir}\nModèle : {model_dir / 'best_model.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
