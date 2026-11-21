"""Lecture des annotations officielles et audit des images sans les modifier."""

import hashlib
from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


SPECIES = {1: "Chat", 2: "Chien"}


def read_annotations(dataset_root):
    """Conserve les identifiants Oxford (1-based) et ajoute notre cible 0/1."""
    dataset_root = Path(dataset_root)
    rows = []
    for split in ("trainval", "test"):
        annotation_path = dataset_root / "annotations" / f"{split}.txt"
        for line_number, line in enumerate(annotation_path.read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{annotation_path}:{line_number} : quatre champs attendus.")
            image_id, class_id, species_id, breed_id = fields
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_[0-9]+", image_id):
                raise ValueError(f"Identifiant d'image invalide : {image_id}")
            class_id, species_id, breed_id = map(int, (class_id, species_id, breed_id))
            if not 1 <= class_id <= 37 or species_id not in SPECIES or breed_id < 1:
                raise ValueError(f"Étiquette invalide pour {image_id}.")
            rows.append({
                "image_id": image_id,
                "relative_path": f"images/{image_id}.jpg",
                "official_split": split,
                "class_id": class_id,
                "breed": image_id.rsplit("_", 1)[0],
                "species_id": species_id,
                "species": SPECIES[species_id],
                "target": species_id - 1,
                "breed_id": breed_id,
            })

    if not rows:
        raise ValueError("Aucune annotation trouvée.")
    frame = pd.DataFrame(rows)
    duplicates = frame.loc[frame.image_id.duplicated(keep=False), "image_id"].tolist()
    if duplicates:
        raise ValueError(f"Identifiants présents plusieurs fois ou dans plusieurs partitions : {duplicates[:5]}")
    for column in ("breed", "species_id", "breed_id"):
        if frame.groupby("class_id")[column].nunique().gt(1).any():
            raise ValueError(f"Correspondance incohérente entre class_id et {column}.")
    if frame.groupby("breed").class_id.nunique().gt(1).any():
        raise ValueError("Une race correspond à plusieurs class_id.")
    return frame


def inspect_image(path):
    """Décode complètement l'image et calcule deux empreintes de doublons exacts."""
    try:
        file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            original_mode = image.mode
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            dimensions = f"{rgb.width}x{rgb.height}:RGB:".encode()
            pixel_hash = hashlib.sha256(dimensions + rgb.tobytes()).hexdigest()
            return {
                "valid": True, "width": rgb.width, "height": rgb.height,
                "original_mode": original_mode,
                "file_sha256": file_hash, "pixel_sha256": pixel_hash, "error": "",
            }
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return {
            "valid": False, "width": None, "height": None,
            "original_mode": "", "file_sha256": "", "pixel_sha256": "",
            "error": f"{type(error).__name__}: {error}",
        }


def find_exact_duplicates(manifest):
    """Les copies détectées partagent les mêmes pixels RGB après orientation EXIF."""
    valid = manifest.loc[manifest.valid].copy()
    duplicates = valid.loc[valid.pixel_sha256.duplicated(keep=False)].copy()
    if duplicates.empty:
        duplicates["cross_official_splits"] = pd.Series(dtype=bool)
    else:
        duplicates["cross_official_splits"] = (
            duplicates.groupby("pixel_sha256").official_split.transform("nunique").gt(1)
        )
    return duplicates.sort_values(["pixel_sha256", "image_id"])


def create_partitions(manifest, seed=42, validation_fraction=0.20):
    """Sépare trainval par race et conserve exactement les membres du test officiel.

    Les anomalies bloquent la création : aucune exclusion n'est implicite.
    Le tri initial rend la séparation indépendante de l'ordre du CSV source.
    """
    required = {
        "image_id", "relative_path", "official_split", "class_id", "species",
        "species_id", "target", "valid", "file_sha256", "pixel_sha256",
    }
    if not required.issubset(manifest.columns):
        raise ValueError(f"Colonnes manquantes : {sorted(required - set(manifest.columns))}")
    if manifest.empty or manifest[list(required)].isna().any().any():
        raise ValueError("Le manifeste est vide ou contient des valeurs obligatoires absentes.")
    if not manifest.valid.eq(True).all():
        raise ValueError("Des images sont invalides : résoudre l'audit avant de séparer.")
    if manifest.image_id.duplicated().any() or manifest.relative_path.duplicated().any():
        raise ValueError("Des identifiants ou chemins sont répétés.")
    if set(manifest.official_split) != {"trainval", "test"}:
        raise ValueError("Les partitions officielles attendues sont trainval et test.")
    for column in ("file_sha256", "pixel_sha256"):
        if not manifest[column].astype(str).str.fullmatch(r"[a-f0-9]{64}").all():
            raise ValueError(f"Empreinte invalide dans {column} : refaire l'audit.")
        if manifest[column].duplicated().any():
            raise ValueError("Doublons exacts détectés : documenter leur traitement avant de séparer.")
    if not manifest.species_id.isin(SPECIES).all():
        raise ValueError("Code d'espèce invalide.")
    if not manifest.target.eq(manifest.species_id - 1).all():
        raise ValueError("La cible ne correspond pas à l'espèce annotée.")
    if not manifest.species.eq(manifest.species_id.map(SPECIES)).all():
        raise ValueError("Le nom de l'espèce ne correspond pas à son code.")
    if manifest.groupby("class_id").species_id.nunique().gt(1).any():
        raise ValueError("Une classe de race correspond à plusieurs espèces.")
    if not 0 < validation_fraction < 1:
        raise ValueError("La fraction de validation doit être comprise entre 0 et 1, exclusivement.")

    ordered = manifest.sort_values("image_id").reset_index(drop=True)
    pool = ordered.loc[ordered.official_split.eq("trainval")]
    train, validation = train_test_split(
        pool, test_size=validation_fraction, random_state=seed, stratify=pool.class_id,
    )
    if set(train.class_id) != set(pool.class_id) or set(validation.class_id) != set(pool.class_id):
        raise ValueError("Chaque race de trainval doit être représentée dans les deux ensembles.")
    partitions = {
        "train": train,
        "validation": validation,
        "test": ordered.loc[ordered.official_split.eq("test")],
    }
    for name, frame in partitions.items():
        partitions[name] = frame.sort_values("image_id").reset_index(drop=True).assign(split=name)
    combined = pd.concat(partitions.values(), ignore_index=True)
    if len(combined) != len(manifest) or combined.image_id.duplicated().any():
        raise ValueError("La couverture des images ou l'unicité entre ensembles est incorrecte.")
    return partitions
