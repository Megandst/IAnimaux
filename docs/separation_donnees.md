# Étape 3 — séparation et chargement des données

## Objectif et résultats

L'entraînement doit apprendre sur des images distinctes de celles utilisées pour
choisir le modèle et mesurer ses performances finales. Les partitions sont
maintenant enregistrées dans `results/splits/`, sans copier ni modifier les photos.

| Ensemble | Chats | Chiens | Total |
|---|---:|---:|---:|
| Entraînement (`train`) | 950 | 1 994 | 2 944 |
| Validation (`validation`) | 238 | 498 | 736 |
| Test officiel (`test`) | 1 183 | 2 486 | 3 669 |
| **Total** | **2 371** | **4 978** | **7 349** |

Les 37 races sont présentes dans chacun des trois ensembles. Les identifiants
du test officiel sont conservés exactement. Aucun identifiant ni aucune empreinte
de pixels ne se répète entre les ensembles. Les empreintes des fichiers actuels
ont été comparées à celles du manifeste d'exploration avant l'enregistrement.

## Comment fonctionne la séparation

`config.py` centralise les chemins, la graine `SEED = 42` et la fraction
`VALIDATION_FRACTION = 0.20`. Le script `scripts/create_splits.py` vérifie les
annotations officielles contre le manifeste, puis utilise `train_test_split`
de scikit-learn, exclusivement sur `trainval` :

```python
train, validation = train_test_split(
    pool,
    test_size=0.20,
    random_state=42,
    stratify=pool.class_id,
)
```

La **stratification** distribue les classes de race de manière aussi proportionnelle
que le permettent les effectifs entiers. Elle évite qu'une race soit absente de
la validation. Le but du modèle reste la classification Chat/Chien ; conserver
les races dans la séparation ne transforme pas ce modèle en classificateur de races.

Le calcul des tailles concerne les 3 680 images de `trainval` :

```text
Validation : 3 680 × 0,20 = 736
Entraînement : 3 680 - 736 = 2 944
Test : 3 669 images officielles conservées
```

Il ne s'agit donc pas d'une séparation 80/20 de l'ensemble complet des 7 349 images.
Le test reste à part.

La **graine aléatoire** initialise le tirage de façon reproductible ; le nombre
42 n'améliore pas les performances. Le tri initial des identifiants rend le résultat
indépendant de l'ordre des lignes du manifeste. Les versions des bibliothèques
et les listes sauvegardées restent nécessaires : une graine seule ne garantit pas
une reproduction identique sur tous les logiciels et tous les matériels.

## Garanties et décisions

- Une image invalide, une cible incohérente ou un doublon exact bloque la création.
  Aucune exclusion automatique n'est pratiquée. L'audit actuel ne nécessite aucune exclusion.
- Les annotations du manifeste doivent correspondre aux fichiers officiels.
- Les empreintes SHA-256 des photos actuelles doivent correspondre à l'audit.
- Le script refuse de remplacer un CSV existant par une séparation différente.
  Pour une expérience alternative justifiée, utiliser un autre `--output-dir`.
- Les fichiers CSV sont triés et leurs empreintes sont enregistrées dans `summary.json`.
- Une seconde exécution sur les données réelles a produit les mêmes CSV octet pour octet.

Le contrôle des doublons porte sur les fichiers et pixels exacts. Il ne prouve pas
l'absence de toutes les ressemblances visuelles entre images.

## Transformations et chargement

`ianimaux/datasets.py` définit `PetImageDataset`. Chaque élément retourné est
un couple `(tenseur_image, cible)` : l'image a trois canaux RGB et la cible
entière vaut 0 pour Chat ou 1 pour Chien. Le rôle enregistré dans le CSV détermine
les transformations, ce qui évite d'activer par erreur les augmentations sur la validation.

| Opération | Entraînement | Validation et test |
|---|---|---|
| Orientation EXIF et conversion RGB | Oui | Oui |
| Petit côté redimensionné à 256 | Oui | Oui |
| Recadrage de 224 × 224 | Aléatoire | Central |
| Retournement horizontal | Probabilité 0,5 | Aucun |
| Conversion en tenseur et normalisation ImageNet | Oui | Oui |

L'**augmentation** crée des variations à la lecture, sans fabriquer de nouveaux
fichiers. Une même photo peut donc être cadrée ou retournée différemment pendant
l'entraînement. L'étiquette reste inchangée. Cela vise à réduire la dépendance
du modèle à un cadrage particulier ; son effet sur les performances sera à évaluer.
Un recadrage peut également retirer une partie de l'animal : cette limite subsiste.

La validation et le test sont déterministes. Deux lectures d'une même image de
validation produisent le même tenseur. Aucune augmentation aléatoire n'y est appliquée.

Le **DataLoader** regroupe les éléments en lots. Le contrôle utilise des lots
de 16 images : `(16, 3, 224, 224)`, soit nombre d'images, canaux, hauteur et largeur.
Les images sont en `float32` et les cibles en `int64`. Cette taille de lot est
un paramètre du contrôle, pas encore un choix d'hyperparamètre d'entraînement.
L'entraînement est mélangé ; la validation conserve son ordre. Le contrôle utilise
`num_workers=0` pour un fonctionnement simple sur macOS et une graine PyTorch fixe.

Le jeu de test est construit à partir du CSV mais ses photographies ne sont pas
chargées par le contrôle des lots. Aucun modèle n'est construit ou entraîné.

## Reproduire et vérifier

Depuis la racine du projet, avec les données et le manifeste d'exploration présents :

```bash
source .venv/bin/activate
python scripts/create_splits.py
python scripts/check_dataloaders.py
python -m unittest discover -s tests -v
```

Les fichiers produits sont :

- `train.csv`, `validation.csv`, `test.csv` : listes d'images et métadonnées.
- `effectifs_especes.csv`, `effectifs_races.csv` : distributions réelles.
- `summary.json` : graine, fraction, versions, effectifs, garanties et empreintes.
- `loader_checks.json` : résultats du contrôle de chargement.

Les tests couvrent la conservation du test officiel, la stratification, la
couverture complète, la reproductibilité malgré un ordre source différent, le
refus des anomalies et le prétraitement. Ils vérifient notamment qu'une image
RGBA devient un tenseur RGB correctement normalisé, que les lectures de validation
sont identiques et que celles d'entraînement peuvent varier.

## Ce que je dois pouvoir expliquer

- Pourquoi le modèle ne doit-il pas apprendre sur le test ?
- À quoi la validation sert-elle, si le test existe déjà ?
- Quelle différence y a-t-il entre un tirage aléatoire et un tirage stratifié ?
- Pourquoi fixer une graine et sauvegarder les listes d'images ?
- Pourquoi augmenter uniquement les données d'entraînement ?
- Que représentent les dimensions `(16, 3, 224, 224)` ?

La prochaine étape sera d'adapter MobileNetV3 Small à deux sorties, de choisir
les paramètres d'un court essai et d'observer les premières pertes d'entraînement
et de validation. Le test ne participera pas à ces choix.
