# IAnimaux

Projet individuel du cours INF1853 — Introduction à l'intelligence artificielle, UQO.

Remise : **8 décembre 2026**. Présentation : **9 décembre 2026**.

L'objectif prioritaire est de classifier des photos de chats et de chiens avec
MobileNetV3 Small, puis de les organiser dans une archive ZIP via Streamlit.
La reconnaissance des races reste facultative.

## État actuel

Étapes 1 à 5 : données préparées, modèle entraîné et interface d'importation
multiple avec prédictions réelles, scores, statistiques, aperçus et corrections
humaines pour le lot courant et le réentraînement. Le classement
dans les dossiers `chat/` et `chien/` est disponible en téléchargement ZIP.
`CONTEXT.md` contient les exigences principales.

## Installation sur macOS

Prérequis : Python 3.11 et un terminal ouvert à la racine du projet.
Vérifier sa présence avec `python3.11 --version`. Si nécessaire, avec Homebrew
déjà installé : `brew install python@3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
python scripts/check_environment.py
```

Dans l'IDE, sélectionner `.venv/bin/python` comme interpréteur.
`requirements-lock.txt` consigne les versions installées sur le Mac ARM64 de
développement pour reproduire l'environnement ; `requirements.txt` liste les
dépendances directes. Une installation sur un autre système devra être vérifiée.

L'installation télécharge les bibliothèques et leurs dépendances. Le téléchargement
Oxford-IIIT Pet est une commande distincte, décrite à l'étape 2. Aucun poids
pré-entraîné n'est nécessaire pour les deux premières étapes.

## Lancer l'application et importer des photos

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Ouvrir l'adresse locale indiquée dans le terminal, habituellement
`http://localhost:8501`. Dans **Importer des photos**, sélectionner plusieurs
fichiers JPG, PNG ou WebP (20 Mo maximum par photo), puis cliquer sur
**Lancer la classification**. Les catégories, scores, compteurs et aperçus
apparaissent sous la progression. Arrêter le serveur avec `Ctrl+C`.

Après vérification des résultats, cliquer sur **Télécharger les photos classées
(ZIP)** dans la carte **Vos photos, bien rangées**. Le fichier
`IAnimaux_photos_classees.zip` contient deux dossiers : `chat/` et `chien/`.
Ouvrir ou extraire le ZIP pour retrouver les photos dans leur catégorie.

Les corrections du lot sont prises en compte. Les noms en double sont adaptés
pour éviter les écrasements, les octets originaux restent inchangés. Les fichiers
illisibles sont exclus et signalés dans l'interface. Le bouton est désactivé si
aucune photo n'a pu être classée. L'archive est générée au clic, en mémoire,
sans créer de dossiers de sortie sur le serveur.

L'export est dans `ianimaux/export.py` et utilise la bibliothèque standard
`zipfile`. Les **47 tests** passent, dont les contrôles des noms, des originaux
et des corrections. Un téléchargement réel dans Chrome a également été vérifié ;
voir `results/interface/verification_telechargement_zip.json`.

Le checkpoint actif `models/refined_feedback_v1/best_model.pt` doit être présent.
Si nécessaire, le fournir depuis les ressources du projet, ou reproduire le
premier entraînement de l'étape 4 puis le réentraînement décrit ci-dessous.
Le chemin actif est défini dans `config.py`. L'application ne télécharge pas
de poids et ne réentraîne pas le modèle.
Les photos importées restent inchangées ; un fichier corrompu n'interrompt pas
le lot. Voir la [notice d'utilisation](docs/utilisation_application.md).

## Apparence de l'interface

L'interface utilise une ambiance ludique pastel (lavande, pêche et menthe),
une illustration de chat et chien et des cartes pour les résultats. La présentation
est adaptée aux petits écrans. Le thème natif est dans `.streamlit/config.toml`,
les styles dans `assets/styles.css` et l'illustration locale dans
`assets/companions.svg`. `ianimaux/ui.py` regroupe l'en-tête et le pied de page.
Inclure ces fichiers lors de la remise du code. Aucun téléchargement d'image
ou de police externe n'est nécessaire à cette présentation.

Les 40 tests passent et le parcours réel d'importation/classification a été
contrôlé dans Chrome sur ordinateur et à une largeur mobile de 390 pixels.
Les captures sont dans `results/interface/apercu_pastel_*.png`.

## Dépannage après une mise à jour

Si Streamlit affiche `ImportError: cannot import name ... from config` alors
que le nom est présent dans `config.py`, arrêter complètement le serveur avec
`Ctrl+C`, puis le relancer :

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Un processus déjà lancé peut conserver d'anciens modules Python en mémoire.
Actualiser le navigateur ne redémarre pas ce processus. Après le redémarrage,
recharger la page et sélectionner à nouveau les photos si nécessaire. Les
corrections enregistrées et les modèles restent conservés sur disque.

## Critères de validation de l'étape 1

- `python -m pip check` ne signale aucune incompatibilité.
- `python scripts/check_environment.py` se termine avec le code 0.
- Le script importe les sept bibliothèques, vérifie un calcul CPU et exécute
  MobileNetV3 Small avec des poids aléatoires, sans téléchargement.
- L'accélération MPS est contrôlée si disponible ; le CPU suffit à cette étape.
- L'écran Streamlit s'affiche dans le navigateur.

Le passage avec des poids aléatoires est un contrôle technique. Il ne démontre
aucune capacité à reconnaître des animaux.

Vérification effectuée le 18 novembre 2026 sur Python 3.11.18, macOS ARM64 :
les imports, le calcul CPU, le passage MobileNetV3 et `pip check` réussissent.
Le contrôle automatisé Streamlit (`AppTest`) exécute l'écran sans exception et
retrouve le titre et le message attendus. L'affichage dans un navigateur reste
à vérifier manuellement avec la commande de lancement ci-dessus.
MPS n'était pas disponible dans l'environnement d'exécution du contrôle.

## Organisation prévue

- `ianimaux/` : modules de données, modèle, prédiction et export.
- `scripts/` : vérification, préparation, entraînement et évaluation.
- `data/` : données locales, exclues de Git.
- `models/` : poids et métadonnées, exclus de Git mais à prévoir dans la remise.
- `results/` : partitions, métriques et graphiques reproductibles.
- `notebooks/` : futurs calculs et illustrations pédagogiques.
- `docs/` : démarche, références et rapport.
- `tests/` : contrôles d'intégrité des données, puis des fonctionnalités principales.

## Explorer les données — étape 2

Le jeu Oxford-IIIT Pet représente environ 800 Mo compressés ; prévoir quelques Go
libres pour les archives et les données extraites. La commande suivante télécharge
les archives officielles via Torchvision et vérifie leurs empreintes MD5 :

```bash
python scripts/prepare_data.py --download
```

Une fois les données présentes, reproduire l'exploration sans réseau :

```bash
python scripts/prepare_data.py
python -m unittest discover -s tests -v
```

Le script lit les annotations d'espèce (1 = chat, 2 = chien), contrôle les
identifiants et les correspondances de races, décode chaque image et recherche
les doublons exacts de pixels, y compris entre les partitions officielles.
Les doublons visuellement proches ne sont pas détectés par cette méthode.
Aucun original n'est modifié. Aucune partition finale ni métrique de modèle
n'est produite à cette étape.

Les résultats sont dans `results/exploration/` :

- `rapport_exploration.md` et `summary.json` : méthode et effectifs réels.
- `manifest.csv` : étiquettes, dimensions, modes, empreintes et erreurs.
- `images_invalides.csv` et `doublons_exacts.csv` : anomalies à examiner avant la séparation.
- `effectifs_especes.csv` et `effectifs_races.csv` : répartition des annotations.
- `effectifs_especes.png` et `dimensions_images.png` : graphiques.
- `exemples_pretraitement.png` : exemples de `trainval` avant/après recadrage.
- `exemples_normalisation.csv` : calculs de normalisation sur des pixels réels.

Les images reproduites dans les figures proviennent d'Oxford-IIIT Pet ; leurs
droits restent ceux des propriétaires d'origine. Voir la licence et les auteurs
sur la page officielle du jeu de données. Ne pas inclure l'ensemble des photos
dans le dépôt Git.

Pour comprendre les calculs et les décisions, lire
`docs/preparation_donnees.md`. Les résultats d'exploration décrivent l'étape 2 ;
les partitions finales sont maintenant enregistrées séparément à l'étape 3.

## Séparer et charger les données — étape 3

```bash
python scripts/create_splits.py
python scripts/check_dataloaders.py
```

Les partitions sont sauvegardées dans `results/splits/` : **2 944 images
d'entraînement**, **736 de validation** et **3 669 de test**. La séparation de
`trainval` est stratifiée par race avec une graine de 42. Les 37 races sont
présentes dans chaque ensemble ; le test officiel est conservé. Les chemins et
ces paramètres sont centralisés dans `config.py`.

Le script contrôle les annotations, les empreintes des photos et l'absence de
doublons avant l'enregistrement. Une séparation existante différente n'est pas
écrasée ; utiliser un autre `--output-dir` pour une expérience alternative.
Une seconde exécution a reproduit les CSV à l'identique.

Le chargeur applique recadrage aléatoire et retournement horizontal uniquement
à l'entraînement. Validation et test utilisent le prétraitement déterministe
officiel. Des lots de 16 images `(16, 3, 224, 224)` ont été vérifiés sur
l'entraînement et la validation ; les photos du test ne sont pas chargées par
ce contrôle. Les résultats sont dans `results/splits/loader_checks.json`.

Les contrôles de données, séparation et prétraitement passent. Aucun
entraînement n'est lancé par ces commandes de préparation. Voir la
[fiche sur la séparation](docs/separation_donnees.md) pour les explications.

## Entraîner le modèle — étape 4

Le premier essai utilise MobileNetV3 Small pré-entraîné sur ImageNet, avec
extracteur et BatchNorm gelés. Le classificateur final à deux sorties apprend
pendant trois époques avec Adam, un taux de 0,001 et des lots de 32. Le premier
chargement télécharge environ 10 Mo de poids officiels. Le benchmark préalable
a estimé environ sept minutes pour trois époques sur CPU avec quatre threads.

```bash
python scripts/train.py --benchmark
python scripts/train.py --epochs 3
```

Chaque expérience conserve son modèle dans `models/<nom>/best_model.pt` et ses
résultats dans `results/training/<nom>/`. Une expérience existante est préservée ;
pour reproduire l'entraînement dans un autre dossier :

```bash
python scripts/train.py --epochs 3 --run-name reproduction_01
```

Le programme enregistre les paramètres, les courbes, les prédictions de validation,
les métriques et le modèle ayant la plus faible perte de validation. Il vérifie
son rechargement. Le test officiel n'est pas utilisé ; les résultats de validation
ne constituent pas encore une mesure finale sur le test.

Les **19 tests** passent, notamment les contrôles de gel de l'extracteur et
de rechargement. Voir la [fiche d'entraînement](docs/entrainement_modele.md).
L'essai `first_run` est terminé en 6 min 44 s : le checkpoint retenu à l'époque 3
atteint **98,23 % d'exactitude et 97,97 % de F1 macro sur la validation**,
contre 67,66 % d'exactitude pour la référence majoritaire. La perte de validation
minimale est 0,0403. Le modèle est sauvegardé et son rechargement vérifié.
Le [résumé de l'expérience](results/training/first_run/summary.json) contient
les résultats réels ; aucune performance sur le test n'est encore disponible.

Les calculs pédagogiques peuvent être reproduits séparément :

```bash
python scripts/explain_learning.py
```

Ces exemples artificiels sont enregistrés dans `results/pedagogie/` et ne
doivent pas être confondus avec les performances du modèle entraîné.

## Interface et inférence — étape 5

`ianimaux/predict.py` applique le même prétraitement déterministe qu'en validation,
puis le modèle sauvegardé en mode évaluation. Aucun seuil de confiance arbitraire
n'est appliqué. Les résultats restent dans la session Streamlit et sont effacés
quand les fichiers sélectionnés changent. Le modèle est mis en cache.

Les **27 tests** passent, dont le parcours d'importation et de classification
avec le checkpoint réel, les fichiers illisibles et la réinitialisation des résultats.
Un contrôle sur deux photos de validation retrouve les probabilités enregistrées
à l'entraînement : `results/interface/verification_inference.json`. Le test
indépendant reste réservé à l'évaluation finale.

## Corrections humaines et réentraînement

Après la classification, ouvrir **Corriger ou confirmer une photo**, choisir
la photo et sa catégorie Chat ou Chien, puis cliquer sur **Enregistrer la correction**.
Le commentaire peut indiquer une race, mais le modèle reste binaire.

La correction rectifie le lot courant. Les nouvelles classifications utilisent
les sorties du modèle, sans réapplication automatique d'une ancienne annotation.
Le tableau conserve séparément
la prédiction et le score IA, la catégorie retenue et l'origine de la décision.
Le bouton de correction ne modifie pas les poids : améliorer d'autres photos
demande un réentraînement contrôlé. Les photos explicitement corrigées sont stockées dans
`data/feedback/corrections.sqlite3` avec un historique et la version du modèle.

La correction Chat de `vraies photos/images (15).jpeg` est déjà enregistrée
d'après le retour utilisateur. Le modèle brut prédit toujours Chien à 54,07 %.
Son recadrage central retire les yeux et les oreilles, un facteur possible de
l'erreur à étudier sans modifier silencieusement le prétraitement évalué.

Pour préparer un nouvel instantané des exemples externes à examiner :

```bash
python scripts/export_feedback.py --output-dir data/feedback/export_02
```

`export_01` contient le premier exemple corrigé. Le script protège les partitions
Oxford connues par empreintes de fichiers et pixels, puis exporte les photos et
un manifeste. `scripts/train.py` ne consomme ces corrections qu'avec l'option
explicite `--feedback-dir`. L'historique, la protection des données réservées
et l'absence de remplacement automatique des prédictions sont testés.
Voir la [fiche sur les corrections humaines](docs/corrections_humaines.md).

Deux stratégies de réentraînement sont comparées : conserver le prétraitement
d'origine ou préserver toute la photo avec des marges. Les modèles et les
résultats restent dans des dossiers séparés, le test officiel n'est pas consulté.
Voir le [protocole de réentraînement](docs/reentrainement_generalisation.md).

Le modèle actif est maintenant **`refined_feedback_v1`** : **99,05 % d'exactitude**,
**98,91 % de F1 macro**, **18 sphynx reconnus sur 20 en validation**, contre
17/20 pour `first_run`. L'ancienne version est conservée. La variante avec marges
atteint 19/20 sur les sphynx mais régresse globalement (97,28 %), elle n'est donc
pas activée. Le test final n'a pas été utilisé.

```bash
python scripts/train.py --run-name refined_feedback_v1 \
  --init-checkpoint models/first_run/best_model.pt \
  --feedback-dir data/feedback/export_01 \
  --learning-rate 0.0001 --epochs 3
python scripts/compare_runs.py --candidate refined_feedback_v1
```

Choisir un autre nom pour reproduire une expérience déjà présente. Le premier
export corrigé est une ressource locale à fournir pour cette reproduction.
Les **40 tests** passent. Les nouvelles classifications ne réappliquent pas
automatiquement une annotation passée ; seule une correction explicite du lot
peut modifier sa catégorie retenue.

## Démarche académique

Le [rapport technique d’avancement](docs/rapport_technique_IAnimaux.md) rassemble
les choix, les réalisations et les résultats vérifiés au 19 novembre 2026.

Voir `docs/demarche_pedagogique.md`. Les exemples de calcul seront distingués des
mesures réellement obtenues. Le rapport aura au maximum 10 pages, références
comprises. L'utilisation de Codex pour la préparation des fichiers et du code
doit être déclarée ; chaque contribution doit être comprise et vérifiée.

## Sources techniques

- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [MobileNetV3 Small — Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- [Installation de Streamlit](https://docs.streamlit.io/get-started/installation/command-line)
