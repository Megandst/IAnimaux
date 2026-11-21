# Étape 4 — comprendre le premier entraînement d'IAnimaux

## Objectif

Adapter un réseau MobileNetV3 Small pré-entraîné sur ImageNet pour distinguer
Chat et Chien, puis vérifier que le modèle apprend sur l'entraînement et donne
des résultats utiles sur la validation. Le test officiel reste réservé à
l'évaluation finale.

Les paramètres et résultats réels sont enregistrés dans
`results/training/first_run/`. Cette fiche explique la méthode ; le rapport
technique rassemble les résultats de l'expérience terminée.

## Ce qui est transféré et ce qui apprend

Le modèle original extrait des caractéristiques visuelles puis produit 1 000
sorties ImageNet. Nous remplaçons la dernière couche linéaire par deux sorties :

```text
Image RGB normalisée (3 × 224 × 224)
                 ↓
Extracteur MobileNetV3 Small pré-entraîné et gelé
                 ↓
Moyenne spatiale → vecteur de 576 caractéristiques
                 ↓
Classificateur : Linear(576, 1024) → Hardswish → Dropout(0,2)
                 ↓
Nouvelle couche : Linear(1024, 2)
                 ↓
Deux valeurs brutes : logit Chat, logit Chien
```

Le bloc classificateur entier est entraîné, y compris sa première couche
linéaire héritée des poids pré-entraînés. Cela représente **592 898 paramètres
entraînables**, sur **1 519 906 paramètres** au total après adaptation binaire.
Les **927 008 paramètres de l'extracteur** sont conservés.

Un paramètre gelé possède `requires_grad=False`. Cela empêche le calcul de son
gradient, mais ne suffit pas pour les statistiques internes des couches BatchNorm.
Nous conservons donc `model.features.eval()` pendant l'entraînement du
classificateur. Le test automatisé vérifie que poids et statistiques de
l'extracteur ne changent pas après une mise à jour.

Le **dropout**, actif pendant l'entraînement du classificateur, désactive
aléatoirement certaines activations. Il est inactif pendant la validation.
Son objectif est de limiter une dépendance excessive à quelques activations.

## Les paramètres du premier essai

| Paramètre | Valeur | Rôle |
|---|---|---|
| Époques | 3 | Trois passages sur les 2 944 images d'entraînement |
| Taille de lot | 32 | Nombre d'images utilisées pour une mise à jour |
| Taux d'apprentissage | 0,001 | Échelle des mises à jour d'Adam |
| Optimiseur | Adam | Mise à jour des paramètres entraînables |
| Fonction de perte | Entropie croisée | Mesure de l'erreur de classification |
| Graine | 42 | Initialisation et tirages reproductibles dans l'environnement fixé |
| Matériel | CPU, 4 threads | Exécution locale sans GPU requis |
| Critère de sélection | Perte de validation minimale | Choix du checkpoint parmi les époques |

Ces valeurs constituent un point de départ, pas des paramètres prouvés optimaux.
L'entropie croisée n'est pas pondérée par classe dans ce premier essai. Le
déséquilibre est suivi avec les métriques par classe, le F1 macro et une référence
majoritaire. Une pondération éventuelle devra être justifiée par la validation.

Un essai préalable traite 3 lots d'entraînement et 2 lots de validation pour
estimer la durée. Ses scores portent sur quelques lots et ne servent pas à
évaluer le modèle. Aucun checkpoint de cet essai n'est réutilisé : l'entraînement
complet redémarre depuis les poids ImageNet et la graine enregistrée.

## Comment une époque fonctionne

Pour chaque lot d'entraînement :

1. Charger les photos, appliquer les augmentations et la normalisation.
2. Remettre les gradients à zéro avec `optimizer.zero_grad()`.
3. Calculer les deux logits avec le réseau.
4. Calculer l'entropie croisée entre logits et bonnes réponses.
5. Calculer les gradients avec `loss.backward()`.
6. Modifier les paramètres du classificateur avec `optimizer.step()`.

Après l'époque, la validation parcourt les 736 images sans augmentation
aléatoire et sans calcul de gradients. Elle ne modifie pas le modèle.
La moyenne des pertes est pondérée par le nombre d'images de chaque lot :

```text
perte de l'époque = somme(perte moyenne du lot × taille du lot) / nombre d'images
```

Cette pondération tient compte du dernier lot de validation, qui contient moins
de 32 images. L'exactitude est le nombre de bonnes réponses divisé par le total.
Le F1 macro donne le même poids aux F1 des deux espèces, malgré leurs effectifs
différents.

L'exactitude d'entraînement est calculée au fil des mises à jour avec dropout
et augmentations. La validation utilise un modèle fixe en fin d'époque et des
images déterministes. Une validation supérieure à l'entraînement n'est donc pas,
à elle seule, la preuve d'une anomalie.

## Calcul illustré de softmax et d'entropie croisée

L'exemple suivant utilise des nombres pédagogiques, pas les résultats du modèle.
Pour les logits `[2, 1]` et une image réellement étiquetée Chat :

```text
p(Chat) = exp(2) / (exp(2) + exp(1)) ≈ 0,731059
p(Chien) ≈ 0,268941
perte = -ln(p(Chat)) ≈ 0,313262
```

Pour une classification à deux classes, l'entropie croisée générale est :

```text
L = -[y_chat × ln(p_chat) + y_chien × ln(p_chien)]
```

La cible Chat correspond à `[1, 0]`, même si PyTorch reçoit simplement l'indice 0.
`CrossEntropyLoss` attend directement les logits et combine les opérations
nécessaires de manière numériquement stable. Il ne faut donc pas appliquer un
softmax avant de lui transmettre les sorties. Le softmax est utilisé séparément
pour obtenir les scores destinés à l'interprétation des prédictions.

## Comprendre une mise à jour par gradient

Dans le même exemple, les dérivées de la perte par rapport aux deux logits sont :

```text
dL/dz_chat = p_chat - 1 ≈ -0,268941
dL/dz_chien = p_chien ≈ 0,268941
```

Si l'on traite les logits comme deux paramètres d'un exercice simplifié, une
descente de gradient de pas 0,1 donnerait :

```text
nouveau z_chat = 2 - 0,1 × (-0,268941) ≈ 2,026894
nouveau z_chien = 1 - 0,1 × 0,268941 ≈ 0,973106
nouvelle perte ≈ 0,299078
```

La perte diminue. Dans le réseau réel, les logits ne sont pas des paramètres
directement ajustés : la rétropropagation applique la règle de dérivation en
chaîne pour obtenir les gradients des poids et biais du classificateur.
Adam utilise ces gradients et leurs moyennes mobiles pour adapter les mises à
jour. L'exercice illustre le sens du gradient sans reproduire exactement Adam.

## Sauvegarde et interprétation

À chaque amélioration de la perte de validation, `best_model.pt` conserve les
poids, les classes dans l'ordre `[Chat, Chien]`, le prétraitement, l'époque et
les paramètres de l'expérience. Le chargement construit le réseau sans poids
ImageNet, puis restaure les paramètres sauvegardés : aucun téléchargement n'est
nécessaire à ce stade.

Le programme recharge le checkpoint et compare ses probabilités à celles
enregistrées sur deux images de validation. Les tests vérifient aussi la
sauvegarde/relecture sur de petites données artificielles.

Les résultats de validation servent à choisir le modèle. Ils ne remplacent pas
une évaluation finale sur le test indépendant. Les scores softmax ne sont pas
des garanties de justesse et aucun seuil de confiance n'est encore sélectionné.

## Commandes et fichiers

Pour refaire les calculs pédagogiques et contrôler leurs relations :

```bash
python scripts/explain_learning.py
```

Le script vérifie une convolution sur une matrice 3 × 3, la relation entre
softmax et gradient de l'entropie croisée, et la diminution de perte dans le
petit exemple de descente de gradient. Il illustre aussi les métriques sur une
matrice artificielle `[[8, 1], [2, 9]]` (lignes réelles, colonnes prédites) :
exactitude 85 %, précision Chat 80 %, rappel Chat 88,89 % et F1 Chat 84,21 %.
Ces valeurs ne décrivent pas le modèle IAnimaux. Les calculs sont enregistrés
séparément dans `results/pedagogie/calculs_illustratifs.json`.

Dans cet exemple de CNN, le filtre est appliqué sans inversion, selon la
convention de corrélation croisée utilisée par `conv2d`.

```bash
source .venv/bin/activate
python scripts/train.py --benchmark
python scripts/train.py --epochs 3
```

Le premier chargement télécharge environ 10 Mo de poids officiels dans
`.cache/torch/`. Un nouvel entraînement doit utiliser un nom différent pour ne
pas écraser les résultats :

```bash
python scripts/train.py --epochs 3 --run-name experience_02
```

L'entraînement sauvegarde :

- `models/first_run/best_model.pt` : modèle sélectionné et métadonnées.
- `results/training/first_run/config.json` : paramètres et empreintes des partitions.
- `history.csv` : perte, exactitude, F1 macro, effectifs et durées par époque.
- `validation_predictions.csv` : prédictions du modèle sélectionné sur la validation.
- `summary.json` : métriques, matrice de confusion, référence majoritaire et contrôle du rechargement.
- `courbes_apprentissage.png` et `matrice_confusion_validation.png` : figures.

Les modèles sont exclus de Git par défaut. Le checkpoint devra être prévu dans
les ressources de remise et de démonstration.

## Ce que je dois pouvoir expliquer

- Quelle différence entre construire un réseau et charger des poids pré-entraînés ?
- Quelles couches sont gelées et quelles couches apprennent ?
- Pourquoi figer aussi le mode des couches BatchNorm ?
- Quelle différence entre un lot, une mise à jour et une époque ?
- Comment les logits, le softmax, la perte et les gradients sont-ils liés ?
- Pourquoi choisir un checkpoint sur la validation et conserver le test à part ?

Sources : [MobileNetV3 Small — Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
et code installé de PyTorch/Torchvision. Les constantes et dimensions correspondent
à la configuration exécutée et enregistrée dans `config.json`.
