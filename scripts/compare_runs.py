"""Compare deux modèles sur la même validation, sans consulter le test final."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT, SPLITS_DIR
from ianimaux.model import load_checkpoint
from ianimaux.predict import classify_uploads


def validation_results(predictions, annotations):
    if predictions.image_id.duplicated().any() or set(predictions.image_id) != set(annotations.image_id):
        raise ValueError("Les prédictions doivent couvrir exactement une fois toute la validation.")
    frame = predictions.merge(annotations[["image_id", "breed", "target"]], on="image_id", validate="one_to_one", suffixes=("", "_expected"))
    if not frame.target.eq(frame.target_expected).all() or not frame.prediction.isin([0, 1]).all():
        raise ValueError("Les étiquettes ne correspondent pas à la validation réservée.")
    frame["correct"] = frame.target.eq(frame.prediction)
    sphynx = frame.loc[frame.breed.eq("Sphynx")]
    if sphynx.empty:
        raise ValueError("Le sous-ensemble Sphynx est absent.")
    probabilities = frame[["p_chat", "p_chien"]]
    sums = probabilities.sum(axis=1)
    if probabilities.isna().any().any() or not probabilities.ge(0).all().all() or not probabilities.le(1).all().all() or not sums.sub(1).abs().le(1e-5).all():
        raise ValueError("Probabilités invalides dans les prédictions.")
    # CSV floats are read as float64, while the network computed softmax in float32.
    probabilities = probabilities.div(sums, axis=0)
    summary = {
        "images": len(frame),
        "accuracy": float(accuracy_score(frame.target, frame.prediction)),
        "f1_macro": float(f1_score(frame.target, frame.prediction, labels=[0, 1], average="macro", zero_division=0)),
        "log_loss": float(log_loss(frame.target, probabilities, labels=[0, 1])),
        "sphynx": {"correct": int(sphynx.correct.sum()), "images": len(sphynx), "recall": float(sphynx.correct.mean())},
        "classification_report": classification_report(frame.target, frame.prediction, labels=[0, 1], target_names=["Chat", "Chien"], output_dict=True, zero_division=0),
    }
    breeds = frame.groupby("breed").correct.agg(["sum", "count"]).rename(columns={"sum": "correct", "count": "images"})
    return summary, breeds


def promotion_checks(baseline, candidate):
    return {
        "macro_f1_not_lower": candidate["f1_macro"] >= baseline["f1_macro"],
        "overall_accuracy_not_lower": candidate["accuracy"] >= baseline["accuracy"],
        "sphynx_recall_strictly_higher": candidate["sphynx"]["recall"] > baseline["sphynx"]["recall"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="first_run")
    parser.add_argument("--candidate", default="full_frame_feedback_v1")
    parser.add_argument("--probe-photo", type=Path, help="Exemple de développement, présenté séparément des mesures de validation.")
    args = parser.parse_args()
    for name in (args.baseline, args.candidate):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("Les noms d'expérience doivent contenir des lettres, chiffres, tirets ou soulignements.")
    annotations = pd.read_csv(SPLITS_DIR / "validation.csv")
    validation_hash = hashlib.sha256((SPLITS_DIR / "validation.csv").read_bytes()).hexdigest()
    summaries, breeds, hashes, configurations = {}, {}, {}, {}
    for name in (args.baseline, args.candidate):
        folder = PROJECT_ROOT / "results" / "training" / name
        settings = json.loads((folder / "config.json").read_text())
        configurations[name] = settings
        if settings["partition_sha256"]["validation"] != validation_hash:
            raise ValueError("L'expérience n'utilise pas la même partition de validation.")
        predictions = pd.read_csv(folder / "validation_predictions.csv")
        summaries[name], breeds[name] = validation_results(predictions, annotations)
        hashes[name] = hashlib.sha256((PROJECT_ROOT / "models" / name / "best_model.pt").read_bytes()).hexdigest()
    initial_path = PROJECT_ROOT / "results" / "training" / args.candidate / "initial_validation_predictions.csv"
    initial = validation_results(pd.read_csv(initial_path), annotations)[0] if initial_path.is_file() else None
    checks = promotion_checks(summaries[args.baseline], summaries[args.candidate])
    preprocessing_changed = configurations[args.baseline].get("preprocessing", "center_crop") != configurations[args.candidate].get("preprocessing", "center_crop")
    summary = {
        "baseline_name": args.baseline, "candidate_name": args.candidate,
        "baseline": summaries[args.baseline], "candidate": summaries[args.candidate],
        "new_preprocessing_before_retraining": initial,
        "promotion_checks": checks, "eligible_for_activation": all(checks.values()),
        "checkpoint_sha256": hashes, "validation_sha256": validation_hash,
        "test_used": False,
        "preprocessing_changed": preprocessing_changed,
        "limitations": "20 Sphynx de validation ; aucune garantie sur toutes les nouvelles photos. Poursuite d'entraînement et ajout de données simultanés ; pas d'attribution isolée à la correction humaine.",
    }
    if args.probe_photo:
        torch.set_num_threads(4)
        probe = {"filename": args.probe_photo.name, "role": "development_example_used_in_candidate_training", "independent_evaluation": False}
        for name in (args.baseline, args.candidate):
            model, _ = load_checkpoint(PROJECT_ROOT / "models" / name / "best_model.pt")
            file = BytesIO(args.probe_photo.read_bytes())
            file.name = args.probe_photo.name
            prediction = classify_uploads(model, [file])[0]
            probe[name] = {"category": prediction["category"], "confidence": prediction["confidence"]}
        summary["development_probe"] = probe
    output = PROJECT_ROOT / "results" / "comparisons" / args.candidate
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    breeds[args.baseline].add_prefix("baseline_").join(breeds[args.candidate].add_prefix("candidate_")).to_csv(output / "per_breed_comparison.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
