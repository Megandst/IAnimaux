"""Fige le modèle avant le test ; ne charge aucune image et ne fait aucune prédiction."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import MODEL_PATH, SEED
from ianimaux.datasets import image_transform
from ianimaux.model import load_checkpoint

DESTINATION = ROOT / 'results' / 'evaluation' / 'binary_final_v1'
SOURCES = ['config.py', 'ianimaux/model.py', 'ianimaux/datasets.py',
           'ianimaux/predict.py', 'scripts/freeze_evaluation.py',
           'results/model_activation.json',
           'results/training/refined_feedback_v1/config.json',
           'results/training/refined_feedback_v1/summary.json',
           'results/splits/summary.json', 'requirements.txt', 'requirements-lock.txt']
PACKAGES = ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(directory=DESTINATION):
    """Vérifie copie, sources, partitions et environnement sans ouvrir les photos."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    failures = []
    for section in ('frozen_files_sha256', 'source_files_sha256', 'partition_sha256'):
        base = directory if section == 'frozen_files_sha256' else ROOT
        for name, expected in manifest[section].items():
            path = base / name
            if not path.is_file() or digest(path) != expected:
                failures.append(f'{section}: {name}')
    if str(MODEL_PATH.relative_to(ROOT)) != manifest['active_checkpoint']:
        failures.append('Chemin du modèle actif modifié')
    if not MODEL_PATH.is_file() or digest(MODEL_PATH) != manifest['checkpoint_sha256']:
        failures.append('Poids du modèle actif modifiés')
    for package, expected in manifest['environment']['packages'].items():
        if version(package) != expected:
            failures.append(f'Version différente : {package}')
    if platform.python_version() != manifest['environment']['python']:
        failures.append('Version Python différente')
    if failures:
        raise ValueError('Gel non conforme :\n' + '\n'.join(failures))
    return manifest


def freeze():
    if DESTINATION.exists():
        raise ValueError('Gel déjà présent : utiliser --verify. Aucun écrasement autorisé.')
    model, checkpoint = load_checkpoint(MODEL_PATH)
    activation = json.loads((ROOT / 'results/model_activation.json').read_text())
    checkpoint_hash = digest(MODEL_PATH)
    if checkpoint_hash != activation['active_sha256']:
        raise ValueError('Le modèle ne correspond pas à la version activée.')
    settings = json.loads((ROOT / 'results/training/refined_feedback_v1/config.json').read_text())
    if settings != checkpoint['settings']:
        raise ValueError('Les paramètres archivés et le checkpoint divergent.')
    splits = json.loads((ROOT / 'results/splits/summary.json').read_text())
    partitions = {}
    for name, expected in splits['partition_sha256'].items():
        relative = f'results/splits/{name}.csv'
        actual = digest(ROOT / relative)  # Octets du CSV seulement, aucune image.
        if actual != expected:
            raise ValueError(f'Partition modifiée : {name}')
        partitions[relative] = actual
    source_hashes = {name: digest(ROOT/name) for name in SOURCES}
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    manifest = {
        'schema_version': 1, 'freeze_id': 'binary_final_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'active_checkpoint': str(MODEL_PATH.relative_to(ROOT)),
        'evaluation_checkpoint': 'results/evaluation/binary_final_v1/best_model.pt',
        'checkpoint_sha256': checkpoint_hash, 'architecture': checkpoint['architecture'],
        'class_names': checkpoint['class_names'], 'selected_epoch': checkpoint['epoch'],
        'preprocessing': checkpoint['preprocessing'],
        'evaluation_transform': repr(image_transform(False, model.preprocessing_name)),
        'training_settings': settings, 'validation_metrics': checkpoint['validation_metrics'],
        'parameters': {'total': sum(p.numel() for p in model.parameters()),
                       'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)},
        'evaluation_protocol': {
            'split': 'test', 'expected_images': splits['counts']['test'],
            'mode': 'eval + torch.inference_mode', 'device': 'cpu',
            'batch_size': 32, 'num_workers': 0, 'cpu_threads': 4, 'seed': SEED,
            'shuffle': False, 'augmentation': False,
            'decision': 'argmax des deux logits ; égalité -> indice 0 (Chat)',
            'confidence': 'softmax, sans seuil de rejet ni calibration',
            'human_correction': False, 'annotation_replay': False,
            'majority_baseline_target_from_train': splits['training_majority_target'],
            'metrics': ['accuracy','precision_per_class','recall_per_class','f1_per_class','f1_macro','confusion_matrix'],
            'selection_on_test': False,
        },
        'test_images_read_by_freeze': False, 'test_inference_performed': False,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: version(name) for name in PACKAGES}},
        'git_head_at_freeze': commit,
        'source_files_sha256': source_hashes, 'partition_sha256': partitions,
    }
    DESTINATION.mkdir(parents=True)
    shutil.copyfile(MODEL_PATH, DESTINATION/'best_model.pt')
    for name in SOURCES:
        target = DESTINATION/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    frozen = [DESTINATION/'best_model.pt'] + list((DESTINATION/'snapshot').rglob('*'))
    manifest['frozen_files_sha256'] = {str(p.relative_to(DESTINATION)):digest(p) for p in frozen if p.is_file()}
    (DESTINATION/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    verify_freeze()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Vérifier le gel existant sans le modifier')
    args = parser.parse_args()
    try:
        manifest = verify_freeze() if args.verify else freeze()
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(f"Gel {'vérifié' if args.verify else 'créé'} : {manifest['freeze_id']}")
    print(f"SHA-256 : {manifest['checkpoint_sha256']}")
    print('Aucune image de test chargée ; aucune prédiction effectuée.')


if __name__ == '__main__':
    main()
