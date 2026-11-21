"""Vérifie l'installation sans télécharger de données ou de modèle."""

import importlib
from importlib import metadata
import platform
import sys


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "streamlit": "streamlit",
    "Pillow": "PIL",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}


def main():
    print("IAnimaux — Vérification de l'environnement")
    print(f"Python : {platform.python_version()}")
    print(f"Interpréteur : {sys.executable}")
    print(f"Système : {platform.system()} ({platform.machine()})")

    if sys.version_info < (3, 11):
        print("[ÉCHEC] Python 3.11 ou supérieur est nécessaire ; 3.11 est la cible du projet.")
        return 1

    failures = []
    modules = {}
    for distribution, module_name in DEPENDENCIES.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            print(f"[OK] {distribution} : {metadata.version(distribution)}")
        except Exception as error:
            failures.append(distribution)
            print(f"[ÉCHEC] {distribution} : {error}")

    if failures:
        print("Installer les dépendances : python -m pip install -r requirements.txt")
        return 1

    torch = modules["torch"]
    try:
        # A small calculation validates execution, not just imports.
        values = torch.tensor([1.0, 2.0, 3.0])
        result = (values * 2).sum().item()
        if result != 12.0:
            raise RuntimeError(f"Résultat CPU inattendu : {result}")
        print("[OK] Calcul CPU : (1 × 2) + (2 × 2) + (3 × 2) = 12")

        # No pretrained weights: this only checks the Torch/Torchvision pairing.
        model = modules["torchvision"].models.mobilenet_v3_small(weights=None)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        if tuple(output.shape) != (1, 1000):
            raise RuntimeError(f"Dimensions inattendues : {tuple(output.shape)}")
        print("[OK] MobileNetV3 Small : passage technique avec des poids aléatoires.")
        print("     Les 1 000 sorties sont celles de l'architecture originale ; aucune prédiction utile ici.")
    except Exception as error:
        print(f"[ÉCHEC] Exécution PyTorch/Torchvision : {error}")
        return 1

    if torch.backends.mps.is_available():
        try:
            result = (values.to("mps") * 2).sum().item()
            if result != 12.0:
                raise RuntimeError(f"Résultat MPS inattendu : {result}")
            print("[OK] Calcul MPS : accélération du Mac disponible.")
        except Exception as error:
            print(f"[INFO] MPS non utilisable pour ce contrôle : {error}")
            print("       Le CPU reste disponible.")
    else:
        print("[INFO] MPS indisponible ; le CPU reste disponible.")

    print("[OK] Vérification terminée. Aucun téléchargement ni entraînement effectué.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
