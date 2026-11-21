"""MobileNetV3 Small binaire : construction et chargement sans dépendance à l'UI."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


CLASS_NAMES = ["Chat", "Chien"]
PREPROCESSING = {
    "resize_short_side": 256,
    "crop_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "color_mode": "RGB",
    "orientation": "EXIF transpose",
}
LETTERBOX_PREPROCESSING = {
    "resize_mode": "letterbox", "size": 224, "padding_rgb": [124, 116, 104],
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "color_mode": "RGB", "orientation": "EXIF transpose",
}


def preprocessing_metadata(name):
    if name == "center_crop":
        return PREPROCESSING
    if name == "letterbox":
        return LETTERBOX_PREPROCESSING
    raise ValueError("Prétraitement inconnu.")


def build_model(pretrained=True):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.features.eval()
    model.preprocessing_name = "center_crop"
    return model


def set_model_mode(model, training):
    model.train(training)
    # Frozen BatchNorm buffers must not drift, even when the classifier is training.
    model.features.eval()


def load_checkpoint(path):
    """Recharge sur CPU, sans téléchargement de poids ImageNet."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Architecture du checkpoint inconnue.")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Classes ou prétraitement incompatibles avec cette version.")
    if checkpoint.get("preprocessing") == PREPROCESSING:
        preprocessing = "center_crop"
    elif checkpoint.get("preprocessing") == LETTERBOX_PREPROCESSING:
        preprocessing = "letterbox"
    else:
        raise ValueError("Prétraitement du checkpoint inconnu.")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.preprocessing_name = preprocessing
    model.eval()
    return model, checkpoint
