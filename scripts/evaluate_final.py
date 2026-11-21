"""Évalue une seule fois la copie gelée sur le test, sans entraînement ni sélection."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

from config import DATASET_ROOT, SPLITS_DIR
from ianimaux.datasets import PetImageDataset, image_transform
from ianimaux.model import load_checkpoint
from scripts.freeze_evaluation import DESTINATION, digest, verify_freeze

OUTPUT = DESTINATION/'test_results'


def summarize_predictions(frame, majority_target):
    """Mesures indépendantes de l'inférence ; les probabilités ne sont pas ajustées."""
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError('Prédictions vides ou identifiants répétés.')
    if not frame.target.isin([0,1]).all() or not frame.prediction.isin([0,1]).all():
        raise ValueError('Cibles ou décisions invalides.')
    scores = frame[['p_chat','p_chien']].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.allclose(scores.sum(1),1,atol=1e-6):
        raise ValueError('Probabilités invalides.')
    if not np.array_equal(scores.argmax(1),frame.prediction.to_numpy()):
        raise ValueError('Décision incompatible avec les scores.')
    if majority_target not in [0,1]:
        raise ValueError('Référence majoritaire invalide.')
    labels = ['Chat','Chien']
    baseline = np.full(len(frame),majority_target)
    return {
        'images':len(frame), 'correct':int(frame.target.eq(frame.prediction).sum()),
        'errors':int(frame.target.ne(frame.prediction).sum()),
        'accuracy':float(accuracy_score(frame.target,frame.prediction)),
        'f1_macro':float(f1_score(frame.target,frame.prediction,labels=[0,1],average='macro',zero_division=0)),
        'classification_report':classification_report(frame.target,frame.prediction,labels=[0,1],target_names=labels,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(frame.target,frame.prediction,labels=[0,1]).tolist(),
        'confusion_matrix_order':labels,
        'majority_baseline':{
            'target_from_train':majority_target,'class':labels[majority_target],
            'accuracy':float(accuracy_score(frame.target,baseline)),
            'f1_macro':float(f1_score(frame.target,baseline,labels=[0,1],average='macro',zero_division=0)),
        },
    }


def make_figures(frame, summary, directory):
    matrix = np.asarray(summary['confusion_matrix'])
    figure, axis = plt.subplots(figsize=(5.6,4.5))
    axis.imshow(matrix,cmap='Purples')
    for i in range(2):
        for j in range(2):
            axis.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=15)
    axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Chat','Chien'],yticklabels=['Chat','Chien'],xlabel='Classe prédite',ylabel='Classe réelle',title='Test indépendant — 3 669 images')
    figure.tight_layout();figure.savefig(directory/'matrice_confusion_test.png',dpi=160);plt.close(figure)
    wrong = frame.loc[frame.target.ne(frame.prediction)].sort_values(['confidence','image_id'],ascending=[False,True])
    # Trois erreurs les plus confiantes et trois les moins confiantes, sans doublon.
    examples = pd.concat([wrong.head(3),wrong.tail(3)]).drop_duplicates('image_id')
    examples.to_csv(directory/'exemples_erreurs.csv',index=False)
    if len(examples):
        figure, axes = plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for axis in axes.flat: axis.axis('off')
        for axis, row in zip(axes.flat,examples.itertuples()):
            with Image.open(DATASET_ROOT/row.relative_path) as original:
                photo = ImageOps.exif_transpose(original).convert('RGB')
                photo.thumbnail((600,450))
                axis.imshow(photo)
            axis.set_title(f'{row.image_id}\nRéel : {row.species} · Prédit : {row.predicted_species}\nScore : {row.confidence:.1%}',fontsize=10)
        figure.suptitle('Erreurs de test : trois scores élevés et trois scores faibles')
        figure.tight_layout();figure.savefig(directory/'exemples_erreurs.png',dpi=130);plt.close(figure)


def main():
    if OUTPUT.exists():
        raise SystemExit('Résultats déjà présents : aucun écrasement ni nouvelle inférence automatique.')
    manifest = verify_freeze()
    protocol = manifest['evaluation_protocol']
    torch.set_num_threads(protocol['cpu_threads'])
    torch.manual_seed(protocol['seed'])
    torch.use_deterministic_algorithms(True)
    model, checkpoint = load_checkpoint(ROOT/manifest['evaluation_checkpoint'])
    if checkpoint['class_names'] != manifest['class_names'] or checkpoint['preprocessing'] != manifest['preprocessing']:
        raise ValueError('Métadonnées de la copie incompatibles avec le gel.')
    dataset = PetImageDataset(SPLITS_DIR/'test.csv',DATASET_ROOT,preprocessing=model.preprocessing_name)
    if len(dataset) != protocol['expected_images'] or dataset.split != 'test':
        raise ValueError('Effectif ou partition inattendus.')
    if repr(image_transform(False,model.preprocessing_name)) != manifest['evaluation_transform']:
        raise ValueError('Transformation différente de celle fixée.')
    OUTPUT.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    (OUTPUT/'execution.json').write_text(json.dumps({
        'status':'running','started_at_utc':started,'checkpoint_sha256':manifest['checkpoint_sha256'],
        'freeze_manifest_sha256':digest(DESTINATION/'manifest.json'),
        'evaluation_script_sha256':digest(Path(__file__)), 'protocol':protocol,
    },ensure_ascii=False,indent=2)+'\n')
    start = perf_counter()
    # Vérifie les octets des images réservées avant toute prédiction ; aucun changement des données.
    for path, expected in zip(dataset.paths,dataset.frame.file_sha256):
        if digest(path) != expected:
            raise ValueError(f'Image modifiée : {path.name}')
    data_integrity_seconds = perf_counter()-start
    print(f'Intégrité vérifiée : {len(dataset)} images de test.',flush=True)
    loader = DataLoader(dataset,batch_size=protocol['batch_size'],shuffle=False,num_workers=protocol['num_workers'])
    model.eval()
    records,loss_sum,offset = [],0.,0
    inference_start = perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            if logits.shape != (len(targets),2) or not torch.isfinite(logits).all():
                raise ValueError('Sorties du modèle invalides.')
            loss_sum += float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
            scores = logits.softmax(1)
            predictions = logits.argmax(1)
            annotations = dataset.frame.iloc[offset:offset+len(targets)]
            for row, target, predicted, probabilities in zip(annotations.itertuples(),targets.tolist(),predictions.tolist(),scores.tolist()):
                records.append({'image_id':row.image_id,'relative_path':row.relative_path,
                    'breed':row.breed,'species':row.species,'target':target,'prediction':predicted,
                    'predicted_species':manifest['class_names'][predicted],
                    'p_chat':probabilities[0],'p_chien':probabilities[1],
                    'confidence':probabilities[predicted]})
            offset += len(targets)
            if offset % 640 == 0 or offset == len(dataset):
                print(f'Évaluation : {offset}/{len(dataset)} images',flush=True)
    inference_seconds = perf_counter()-inference_start
    frame = pd.DataFrame(records)
    if frame.image_id.tolist() != dataset.frame.image_id.tolist() or frame.target.tolist() != dataset.frame.target.tolist():
        raise ValueError('Couverture ou alignement incorrect du test.')
    # Confirme qu'aucune mutation de poids, du code gelé ou des partitions n'a eu lieu.
    verify_freeze()
    summary = summarize_predictions(frame,protocol['majority_baseline_target_from_train'])
    summary.update({'loss':loss_sum/len(frame),'inference_seconds':inference_seconds,
        'data_integrity_seconds':data_integrity_seconds,'started_at_utc':started,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'freeze_id':manifest['freeze_id'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'test_used':True,
        'model_updated':False,'human_corrections_applied':False,'test_used_for_selection':False})
    frame.to_csv(OUTPUT/'predictions.csv',index=False)
    errors = frame.loc[frame.target.ne(frame.prediction)].sort_values(['confidence','image_id'],ascending=[False,True])
    errors.to_csv(OUTPUT/'erreurs.csv',index=False)
    pd.DataFrame(summary['classification_report']).T.to_csv(OUTPUT/'mesures_par_classe.csv')
    pd.DataFrame(summary['confusion_matrix'],index=['Chat','Chien'],columns=['Chat','Chien']).to_csv(OUTPUT/'matrice_confusion.csv')
    # Sous-groupes descriptifs d'espèce : il ne s'agit pas de prédictions de races.
    by_breed = frame.assign(correct=frame.target.eq(frame.prediction)).groupby(['breed','species']).correct.agg(['sum','count'])
    by_breed['accuracy'] = by_breed['sum']/by_breed['count']
    by_breed.to_csv(OUTPUT/'classification_binaire_par_race.csv')
    make_figures(frame,summary,OUTPUT)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    execution = json.loads((OUTPUT/'execution.json').read_text())
    execution.update({'status':'completed','completed_at_utc':summary['completed_at_utc'],
        'output_sha256':{p.name:digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!='execution.json'}})
    (OUTPUT/'execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ['images','correct','errors','accuracy','f1_macro','confusion_matrix','majority_baseline','inference_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
