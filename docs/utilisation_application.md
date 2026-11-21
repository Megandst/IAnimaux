# Étape 5 — importer et classifier des photos avec IAnimaux

## Lancer l'application

La présentation est ludique et pastel : illustration chat/chien en haut,
carte blanche **Vos photos à classer**, puis bilan coloré et aperçus. Sur un
petit écran, les éléments se placent verticalement.

Dans un terminal ouvert à la racine du projet :

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Ouvrir l'adresse locale indiquée par Streamlit, habituellement
`http://localhost:8501`. Si l'application est déjà ouverte pendant une mise à jour,
actualiser la page ou relancer cette commande. Le modèle entraîné doit être
présent dans `models/refined_feedback_v1/best_model.pt`, chemin actif défini dans
`config.py`. Le modèle `first_run` est conservé comme référence historique.

## Importer et analyser

1. Repérer la zone **Importer des photos** en haut de la page.
2. Cliquer sur le bouton de sélection de fichiers de cette zone, ou y déposer les photos.
3. Sélectionner plusieurs fichiers. Sur Mac, maintenir Commande pour sélectionner
   des fichiers séparés, ou Maj pour sélectionner une plage.
4. Cliquer sur **Lancer la classification**.
5. Consulter la progression, les compteurs, le tableau et les aperçus sous les résultats.
6. Corriger les éventuelles erreurs dans le lot.
7. Cliquer sur **Télécharger les photos classées (ZIP)** et ouvrir l'archive obtenue.

Les formats acceptés sont JPG/JPEG, PNG et WebP, jusqu'à **20 Mo par photo**.
Chaque photo doit contenir un animal principal, chat ou chien. Un fichier illisible
est signalé dans le tableau et ne bloque pas les autres fichiers du lot.

Les compteurs affichent les photos classées, les chats, les chiens et les fichiers
en erreur. Les catégories sont des prédictions du modèle. Le score de confiance
provient du softmax et ne garantit pas que la prédiction soit correcte.
Aucun seuil arbitraire de rejet n'est appliqué.

Changer ou retirer les photos efface les anciens résultats. Deux fichiers portant
le même nom restent deux entrées distinctes. Les photos originales ne sont pas
modifiées. Les importations et aperçus restent en mémoire, sauf si une correction
est explicitement enregistrée : la photo et son étiquette humaine sont alors
conservées localement dans `data/feedback/corrections.sqlite3`.

## Corriger une prédiction

Sous le tableau, ouvrir **Corriger ou confirmer une photo**, sélectionner le
fichier, choisir Chat ou Chien et cliquer sur **Enregistrer la correction**.
Un commentaire facultatif peut indiquer « sphynx », sans créer de prédiction de race.

La catégorie retenue suit cette correction dans le lot en cours.
La **Prédiction IA** et la **Confiance IA** restent les sorties originales du
modèle ; **Origine** précise si une décision humaine est utilisée. Les compteurs
suivent la catégorie retenue. L'annotation persiste pour le réentraînement, mais
une nouvelle classification repart des sorties du modèle. Les poids ne sont
pas modifiés par le bouton de correction ; ils évoluent lors d'un réentraînement
explicite, suivi d'une comparaison sur la validation.
Voir [les explications et le cas du sphynx](corrections_humaines.md).

## Télécharger les dossiers chat et chien

La carte **Vos photos, bien rangées** contient le bouton **Télécharger les photos
classées (ZIP)**. Le navigateur télécharge `IAnimaux_photos_classees.zip` à
l'emplacement habituel de ses téléchargements. Ouvrir ou extraire cette archive
pour obtenir :

```text
IAnimaux_photos_classees.zip
├── chat/
│   └── photos classées Chat
└── chien/
    └── photos classées Chien
```

Les deux dossiers sont présents même si l'un est vide. Les corrections explicites
du lot courant déterminent la destination ; les annotations historiques ne sont
pas appliquées à une nouvelle classification. En cas de noms répétés dans un
même dossier, un suffixe est ajouté (`photo.jpg`, `photo_2.jpg`, etc.), y compris
pour les noms ne différant que par leur casse. Les photos ne sont pas réencodées.

Les fichiers illisibles sont exclus du ZIP et leur nombre est affiché. Si tous
les fichiers sont invalides, le bouton est désactivé. Une nouvelle classification
ou correction produit un téléchargement correspondant aux décisions actuelles.
L'archive est créée au clic, en mémoire, sans écrire de copies sur le serveur.

Les 47 tests passent. Un téléchargement effectif a été réalisé dans Chrome avec
deux photos et son ZIP a été ouvert et vérifié : chemins attendus, intégrité CRC
et octets identiques aux fichiers originaux. Le relevé est dans
`results/interface/verification_telechargement_zip.json`.

## Comment fonctionne l'inférence

L'entraînement modifie les paramètres du classificateur. L'inférence les utilise
pour une nouvelle photo, sans les modifier :

```text
Octets du fichier importé
          ↓
Vérification Pillow, orientation EXIF et conversion RGB
          ↓
Redimensionnement, recadrage central et normalisation
          ↓
MobileNetV3 Small rechargé depuis best_model.pt
          ↓
Deux logits → softmax → classe au score maximal
          ↓
Résultat et aperçu dans Streamlit
```

`ianimaux/predict.py` contient ce traitement indépendamment de l'interface.
Le prétraitement est exactement celui de la validation, sans augmentation
aléatoire. `model.eval()` désactive le dropout et `torch.inference_mode()`
désactive le suivi des gradients. Aucun entraînement ni appel à un service
externe de classification n'est effectué.

Streamlit garde le modèle en cache pour éviter de le recharger à chaque
interaction. La date de modification et la taille du checkpoint font partie de
la clé du cache. Les résultats appartiennent à la session et une empreinte des
noms et contenus importés permet de les invalider lorsque la sélection change.

Les aperçus sont réduits à un maximum de 480 × 480 pixels, en conservant les
proportions. Cela n'affecte pas la prédiction, qui utilise l'image avant cette
réduction. Les noms importés ne sont jamais utilisés comme chemins de sortie.

## Vérifications réalisées

Les tests couvrent les données, les partitions, les transformations,
l'entraînement, l'inférence, l'interface et les corrections. Les contrôles de
l'importation et de la prédiction portent notamment sur :

- Une photo corrompue entre deux photos valides, sans interruption du lot.
- La conservation des octets originaux et de la position de lecture du fichier.
- La conversion RGBA/RGB et l'application de l'orientation EXIF.
- La distinction de contenus différents portant le même nom.
- Le refus d'un contenu vide ou d'un format réellement non pris en charge.
- Le bouton désactivé sans sélection et le message de modèle absent.
- L'importation et la classification dans Streamlit avec le checkpoint réel.
- La conservation des résultats lors d'une réexécution, puis leur effacement
  quand les fichiers changent ou sont retirés.

Le parcours Streamlit est testé avec `AppTest`, qui manipule les widgets et exécute
le code de l'application. Il est complété par une revue dans Chrome sur ordinateur
et au format mobile de 390 pixels, avec importation et classification réelles.
Les tests d'interface utilisant le vrai checkpoint sont signalés comme
ignorés si ce fichier n'est pas disponible sur une autre installation.

Une vérification complémentaire a comparé les probabilités d'inférence aux
prédictions de validation enregistrées pour `Abyssinian_10` et
`american_bulldog_103`. Elles correspondent aux tolérances numériques prévues.
Le résultat est dans `results/interface/verification_inference.json`.
Ce contrôle porte sur la cohérence du pipeline, pas sur une nouvelle mesure
globale de performance. Aucune photographie de test n'a été utilisée.

Commande de vérification :

```bash
python -m unittest discover -s tests -v
```

## Ce que je dois pouvoir expliquer

- Quelle différence entre entraînement et inférence ?
- Pourquoi utiliser le même prétraitement qu'en validation ?
- Pourquoi ne pas appliquer d'augmentations aléatoires aux photos importées ?
- Comment les deux logits deviennent-ils une catégorie et un score ?
- Pourquoi le modèle doit-il rester en mode évaluation ?
- Pourquoi un score élevé peut-il accompagner une erreur ?
