"""Explore Oxford-IIIT Pet ; ne crée pas encore les partitions finales."""

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# Keep generated caches within the project, including in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

from ianimaux.data import find_exact_duplicates, inspect_image, read_annotations


def create_figures(manifest, output_dir, dataset_root):
    valid = manifest.loc[manifest.valid]
    counts = manifest.groupby(["official_split", "species"]).size().unstack(fill_value=0)
    axis = counts.plot.bar(rot=0, color=["#578d67", "#ca824a"])
    axis.set(xlabel="Partition officielle", ylabel="Nombre d'images", title="Oxford-IIIT Pet — annotations par espèce")
    axis.legend(title="Espèce")
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "effectifs_especes.png", dpi=150)
    plt.close(axis.figure)

    if valid.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(valid.width, bins=30, alpha=0.65, label="Largeur")
    axis.hist(valid.height, bins=30, alpha=0.65, label="Hauteur")
    axis.set(xlabel="Pixels (orientation EXIF appliquée)", ylabel="Nombre d'images", title="Dimensions des images lisibles")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dimensions_images.png", dpi=150)
    plt.close(figure)

    # Use trainval examples only: no visual selection based on test images.
    examples = valid.loc[valid.official_split.eq("trainval")].groupby("species", sort=True).head(2)
    if examples.empty:
        return
    figure, axes = plt.subplots(len(examples), 2, figsize=(8, 3 * len(examples)), squeeze=False)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()  # This does not download model weights.
    mean = pd.Series(transform.mean).to_numpy()
    std = pd.Series(transform.std).to_numpy()
    pixel_examples = []
    for row_index, row in enumerate(examples.itertuples()):
        with Image.open(dataset_root / row.relative_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            tensor = transform(rgb)
            normalized = tensor.permute(1, 2, 0).numpy()
            # Undo normalization for display; a normalized tensor is not a normal photo.
            display = (normalized * std + mean).clip(0, 1)
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(f"{row.species} — {row.image_id}\nOriginal : {rgb.width} × {rgb.height}")
            axes[row_index, 1].imshow(display)
            axes[row_index, 1].set_title("Entrée : 224 × 224\nNormalisation annulée pour l'affichage")
            for channel, value, channel_mean, channel_std in zip("RGB", tensor[:, 112, 112].tolist(), mean, std):
                scaled = value * channel_std + channel_mean
                pixel_examples.append({
                    "image_id": row.image_id, "channel": channel,
                    "x": 112, "y": 112, "pixel_0_255": round(scaled * 255),
                    "pixel_0_1": scaled, "mean": channel_mean, "std": channel_std,
                    "normalized_value": value,
                })
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "exemples_pretraitement.png", dpi=120)
    plt.close(figure)
    pd.DataFrame(pixel_examples).to_csv(output_dir / "exemples_normalisation.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Télécharger les archives officielles si absentes (~800 Mo).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "exploration")
    args = parser.parse_args()
    dataset_root = args.data_dir / "oxford-iiit-pet"
    if args.download:
        print("Téléchargement officiel via Torchvision (contrôle MD5 des archives).", flush=True)
        OxfordIIITPet(root=args.data_dir, split="trainval", download=True)
    if not (dataset_root / "annotations" / "trainval.txt").is_file():
        parser.error("Données absentes. Relancer avec --download pour les télécharger.")

    annotations = read_annotations(dataset_root)
    print(f"{len(annotations)} annotations lues. Vérification complète des images…", flush=True)
    inspections = []
    for index, row in enumerate(annotations.itertuples(), 1):
        inspections.append(inspect_image(dataset_root / row.relative_path))
        if index % 500 == 0 or index == len(annotations):
            print(f"Images vérifiées : {index}/{len(annotations)}", flush=True)
    manifest = pd.concat([annotations, pd.DataFrame(inspections)], axis=1)
    duplicates = find_exact_duplicates(manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    manifest.loc[~manifest.valid].to_csv(output_dir / "images_invalides.csv", index=False)
    duplicates.to_csv(output_dir / "doublons_exacts.csv", index=False)
    manifest.groupby(["official_split", "species"]).size().rename("count").to_csv(output_dir / "effectifs_especes.csv")
    manifest.groupby(["class_id", "breed", "species", "official_split"]).size().rename("count").to_csv(output_dir / "effectifs_races.csv")
    summary = {
        "annotated_images": len(manifest),
        "valid_images": int(manifest.valid.sum()),
        "invalid_images": int((~manifest.valid).sum()),
        "breeds": int(manifest.class_id.nunique()),
        "species_counts": {key: int(value) for key, value in manifest.species.value_counts().items()},
        "official_split_counts": {key: int(value) for key, value in manifest.official_split.value_counts().items()},
        "original_modes": {key: int(value) for key, value in manifest.loc[manifest.valid, "original_mode"].value_counts().items()},
        "exact_duplicate_groups": int(duplicates.pixel_sha256.nunique()),
        "images_in_exact_duplicate_groups": len(duplicates),
        "cross_official_split_duplicate_groups": int(duplicates.loc[duplicates.cross_official_splits, "pixel_sha256"].nunique()),
        "label_mapping": {"0": "Chat", "1": "Chien"},
        "final_partitions_created": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    create_figures(manifest, output_dir, dataset_root)
    report = f"""# IAnimaux — exploration réelle d'Oxford-IIIT Pet

Commande de reproduction : `python scripts/prepare_data.py` (ajouter `--download` à la première exécution).

- Images annotées : {summary['annotated_images']}.
- Images lisibles : {summary['valid_images']} ; illisibles ou absentes : {summary['invalid_images']}.
- Races : {summary['breeds']}.
- Effectifs par espèce : {summary['species_counts']}.
- Partitions officielles : {summary['official_split_counts']}.
- Groupes de doublons exacts de pixels : {summary['exact_duplicate_groups']}.
- Groupes traversant les partitions officielles : {summary['cross_official_split_duplicate_groups']}.

## Méthode

Les espèces sont lues dans la troisième colonne des annotations officielles :
1 = chat, 2 = chien. La cible du modèle est 0 = Chat, 1 = Chien.
Les noms de fichiers servent à identifier les races, pas à déduire l'espèce.
L'unicité des identifiants et la cohérence des étiquettes sont contrôlées.
Chaque image est vérifiée puis entièrement décodée ; aucune image originale n'est modifiée.
Deux empreintes SHA-256 sont enregistrées : fichier et pixels RGB après orientation EXIF.
La détection de doublons couvre les pixels identiques, pas les photos presque identiques.

## Prétraitement illustré

Le prétraitement officiel des poids MobileNetV3 Small IMAGENET1K_V1 est utilisé :
redimensionnement du petit côté à 256, recadrage central 224 × 224, valeurs divisées
par 255, puis normalisation de chaque canal : `z = (pixel / 255 - moyenne) / écart_type`.
Moyennes RGB : [0.485, 0.456, 0.406] ; écarts-types : [0.229, 0.224, 0.225].
Les calculs sur des pixels réels sont dans `exemples_normalisation.csv`.
Les figures utilisent uniquement des exemples de `trainval` et annulent la
normalisation pour afficher une image interprétable. Le recadrage peut retirer
une partie de l'animal : c'est une limite à examiner.
Aucun poids pré-entraîné n'est téléchargé pour ces transformations.

## Interprétation et prochaine étape

Le nombre de races par espèce contribue au déséquilibre des classes : il faut
rapporter les métriques par classe et le F1 macro, pas seulement l'exactitude.
Les effectifs ci-dessus sont ceux des annotations, avant toute exclusion.
Consulter `images_invalides.csv` et `doublons_exacts.csv` avant de figer les partitions.
Une image illisible ne sera pas silencieusement utilisée pour l'entraînement.
Les éventuels doublons ne devront pas traverser entraînement, validation et test.
La politique d'exclusion sera documentée lors de la création des partitions.
Le test officiel reste réservé à l'évaluation finale ; aucune prédiction ni
sélection de modèle n'est effectuée ici. Les partitions finales ne sont pas encore créées.

Source : [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Les images restent attribuées à leurs propriétaires d'origine ; voir la licence
indiquée sur la page officielle avant toute redistribution.
"""
    (output_dir / "rapport_exploration.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Rapports et figures : {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
