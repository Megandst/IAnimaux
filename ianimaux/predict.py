"""Inférence sur des photos importées, sans modifier les fichiers d'origine."""

import hashlib
from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from config import MAX_UPLOAD_MB
from ianimaux.datasets import image_transform
from ianimaux.model import CLASS_NAMES


class InvalidPhotoError(ValueError):
    """Une photo ne peut pas être préparée pour le modèle."""


def upload_signature(files):
    """Identifie les noms, contenus et ordre du lot, même pour des noms répétés."""
    return tuple((file.name, hashlib.sha256(file.getvalue()).hexdigest()) for file in files)


def read_photo(content):
    if not content:
        raise InvalidPhotoError("Le fichier est vide.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidPhotoError(f"La photo dépasse la limite de {MAX_UPLOAD_MB} Mo.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidPhotoError("Format non pris en charge : utilisez JPG, PNG ou WebP.")
                image.verify()
            with Image.open(BytesIO(content)) as image:
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidPhotoError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise InvalidPhotoError("Le fichier est illisible ou endommagé.") from error
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InvalidPhotoError("Les dimensions de cette image sont trop grandes pour être analysées.") from error


def classify_uploads(model, files, on_progress=None):
    """Une entrée = une ligne de résultat ; une erreur de photo n'arrête pas le lot."""
    transform = image_transform(training=False, preprocessing=getattr(model, "preprocessing_name", "center_crop"))
    results = []
    model.eval()
    for index, file in enumerate(files, 1):
        content = file.getvalue()
        row = {
            "filename": file.name, "category": None, "confidence": None,
            "p_chat": None, "p_chien": None, "error": "", "preview": None,
            "image_sha256": hashlib.sha256(content).hexdigest(),
        }
        try:
            image = read_photo(content)
        except InvalidPhotoError as error:
            row["error"] = str(error)
        else:
            with torch.inference_mode():
                logits = model(transform(image).unsqueeze(0))
                if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                    raise RuntimeError("Le modèle a retourné des sorties invalides.")
                probabilities = logits.softmax(dim=1)[0]
            prediction = int(probabilities.argmax())
            row.update({
                "category": CLASS_NAMES[prediction], "confidence": float(probabilities[prediction]),
                "p_chat": float(probabilities[0]), "p_chien": float(probabilities[1]),
            })
            image.thumbnail((480, 480))
            preview = BytesIO()
            image.save(preview, format="PNG")
            row["preview"] = preview.getvalue()
        results.append(row)
        if on_progress is not None:
            on_progress(index, len(files))
    return results
