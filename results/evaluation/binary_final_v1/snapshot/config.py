"""Chemins et paramètres communs de préparation des données."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "oxford-iiit-pet"
EXPLORATION_DIR = PROJECT_ROOT / "results" / "exploration"
SPLITS_DIR = PROJECT_ROOT / "results" / "splits"
SEED = 42
VALIDATION_FRACTION = 0.20
MODEL_PATH = PROJECT_ROOT / "models" / "refined_feedback_v1" / "best_model.pt"
MAX_UPLOAD_MB = 20
FEEDBACK_DB = PROJECT_ROOT / "data" / "feedback" / "corrections.sqlite3"
