"""Chargement des partitions sauvegardées et transformations propres à leur rôle."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms import InterpolationMode


class FitInsideSquare:
    """Conserve toute la photo et ses proportions, puis complète avec des marges."""

    def __call__(self, image):
        resized = ImageOps.contain(image, (224, 224), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (224, 224), (124, 116, 104))
        canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
        return canvas


def image_transform(training, preprocessing="center_crop"):
    """Augmentations pour train ; prétraitement déterministe pour validation/test."""
    evaluation = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    if preprocessing == "letterbox":
        operations = [FitInsideSquare()]
        if training:
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        return transforms.Compose(operations + [
            transforms.ToTensor(),
            transforms.Normalize(mean=evaluation.mean, std=evaluation.std),
        ])
    if preprocessing != "center_crop":
        raise ValueError("Prétraitement inconnu.")
    if not training:
        return evaluation
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=evaluation.mean, std=evaluation.std),
    ])


class PetImageDataset(Dataset):
    """Retourne (image normalisée, cible entière) sans modifier les photos sources."""

    def __init__(self, csv_path, dataset_root, preprocessing="center_crop"):
        self.frame = pd.read_csv(csv_path)
        required = {"relative_path", "target", "split"}
        if not required.issubset(self.frame.columns) or self.frame.empty:
            raise ValueError("Partition vide ou colonnes absentes.")
        splits = self.frame.split.unique()
        if len(splits) != 1 or splits[0] not in {"train", "validation", "test"}:
            raise ValueError("Une seule partition connue est attendue par fichier.")
        if not self.frame.target.isin([0, 1]).all():
            raise ValueError("Les cibles doivent être 0 (Chat) ou 1 (Chien).")
        self.split = splits[0]
        self.dataset_root = Path(dataset_root).resolve()
        self.transform = image_transform(training=self.split == "train", preprocessing=preprocessing)
        self.paths = []
        for relative_path in self.frame.relative_path:
            if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
                raise ValueError("Un chemin relatif d'image est attendu.")
            path = (self.dataset_root / relative_path).resolve()
            if not path.is_relative_to(self.dataset_root):
                raise ValueError("Le chemin d'image sort du dossier des données.")
            self.paths.append(path)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(self.frame.iloc[index].target)
