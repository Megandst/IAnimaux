# CONTEXT.md — IAnimaux

## 1. Contexte du projet

Nous développons un projet final pour le cours **INF1853 – Introduction à l'intelligence artificielle**, à l'Université du Québec en Outaouais (UQO), session d'automne 2026.

**Enseignant :** Jean Jarcke Mukombelwa Malasi  
**Équipe :** Megan Daoust et Luzia Coelho Bento Muaquesse (deux coauteures ; mise à jour explicite du 21 novembre 2026)<br>
**Date de remise :** 8 décembre 2026  
**Présentation orale :** 9 décembre 2026  
**Pondération :** 25 % de la note finale du cours

Le projet doit démontrer l'utilisation concrète d'une technique d'intelligence artificielle pour résoudre un problème réel.

L'enseignant privilégie un système simple, fonctionnel, bien documenté et correctement évalué plutôt qu'un système trop ambitieux et incomplet.

Nous débutons dans le développement de projets d'intelligence artificielle. Codex doit donc nous guider, expliquer ses décisions techniques et privilégier des solutions pédagogiques, simples et compréhensibles.

**Langue :** les explications, l'interface, le README et le rapport doivent être en français. Le code Python, les noms de variables et les commentaires techniques peuvent être en anglais.

---

## 2. Nom et description du projet

**Nom :** IAnimaux

**Titre officiel :** IAnimaux – Système intelligent de classification automatique de photos d'animaux

**Domaines :**
- Apprentissage automatique (Machine Learning)
- Vision par ordinateur (Computer Vision)
- Classification d'images (Image Classification)

### Problématique

Les refuges animaliers doivent souvent gérer un grand nombre de photographies de chats et de chiens.

Le classement manuel de ces photographies dans différents dossiers est une tâche répétitive qui demande du temps.

Notre objectif est de développer une application utilisant l'intelligence artificielle pour automatiser ce processus.

L'employé d'un refuge pourra importer un ensemble de photographies mélangées de chats et de chiens.

L'application analysera chaque image, prédira s'il s'agit d'un chat ou d'un chien, puis organisera automatiquement les images dans des dossiers distincts.

Une extension facultative permettra également d'identifier certaines races d'animaux.

---

## 3. Objectifs

### Objectif principal — Obligatoire

Développer une application capable de :

1. Importer plusieurs photographies d'animaux.
2. Analyser automatiquement chaque photographie à l'aide d'un modèle d'IA.
3. Reconnaître si l'animal est un chat ou un chien.
4. Afficher la catégorie prédite et son score de confiance.
5. Classer les photographies dans deux dossiers : `chat/` et `chien/`.
6. Fournir un résumé des résultats de classification.
7. Évaluer objectivement les performances du modèle.

### Objectif secondaire — Facultatif

Ajouter la reconnaissance des races.

Exemples de races :

Chats :
- Siamois
- Persan
- Bengal
- Maine Coon

Chiens :
- Golden Retriever
- Beagle
- Labrador Retriever
- Chihuahua

Le modèle pourra utiliser les 37 catégories du jeu de données Oxford-IIIT Pet.

Si cette fonctionnalité est développée, l'application pourra également créer des sous-dossiers par race.

**Important :** la reconnaissance des races ne doit jamais compromettre la réalisation de l'objectif principal.

---

## 4. Fonctionnement attendu

### Scénario d'utilisation

Un employé du refuge possède un dossier contenant 100 photographies mélangées.

Il ouvre IAnimaux, importe les photographies et lance la classification.

Le système analyse les photographies et produit les résultats.

Exemple fictif :

- 100 images analysées.
- 42 images classées comme chats.
- 58 images classées comme chiens.

L'application affiche les résultats et permet de télécharger les images organisées dans une archive ZIP.

### Structure des fichiers générés

Version principale :

```text
photos_classees/
├── chat/
│   ├── chat_001.jpg
│   ├── chat_002.jpg
│   └── chat_003.jpg
└── chien/
    ├── chien_001.jpg
    ├── chien_002.jpg
    └── chien_003.jpg
```

Version avec reconnaissance des races :

```text
photos_classees/
├── chat/
│   ├── Siamois/
│   ├── Persan/
│   └── Autres/
└── chien/
    ├── Golden_Retriever/
    ├── Beagle/
    └── Autres/
```

Une catégorie `A_verifier/` pourra être ajoutée pour les images dont la classification est incertaine ou qui ne peuvent pas être analysées.

L'application doit préserver les fichiers originaux et ne pas écraser ou supprimer les photographies importées.

### Limitation initiale

Dans la première version, chaque photographie doit contenir un animal principal.

La reconnaissance de plusieurs animaux différents dans une seule photographie est hors du périmètre initial.

---

## 5. Choix des technologies

Nous avons considéré deux approches :

1. Weka
2. Python

### Décision : Python

Nous retenons Python, car il est mieux adapté à la vision par ordinateur, à l'utilisation des réseaux neuronaux et à la construction d'une application simple.

Weka est excellent pour expérimenter avec des algorithmes classiques sur des données tabulaires, mais nécessiterait des étapes supplémentaires pour traiter directement des photographies.

Weka pourra éventuellement être utilisé pour une comparaison expérimentale, mais il ne constitue pas notre technologie principale.

### Technologies retenues

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| PyTorch | Apprentissage automatique |
| Torchvision | Modèles pré-entraînés et transformations d'images |
| MobileNetV3 | Modèle de classification d'images |
| Streamlit | Interface graphique |
| Pillow | Lecture et traitement des images |
| scikit-learn | Mesures d'évaluation |
| Matplotlib | Visualisation des résultats |
| Pandas | Organisation des résultats |

Nous devons éviter les dépendances inutiles et garder l'installation simple.

---

## 6. Approche d'intelligence artificielle

### Technique principale

Nous utiliserons un **réseau de neurones convolutif (CNN)** avec une technique d'**apprentissage par transfert (Transfer Learning)**.

### Modèle retenu

**MobileNetV3**, idéalement la variante Small si cela simplifie l'entraînement et l'exécution sur un ordinateur personnel.

Ce modèle est relativement léger et adapté à la classification d'images.

### Principe

Un CNN analyse les caractéristiques visuelles des photographies.

Au lieu d'entraîner un réseau entièrement à partir de zéro, nous utiliserons un modèle pré-entraîné sur un grand jeu de données d'images.

Nous adapterons sa couche de classification pour reconnaître les catégories de notre projet.

Pour l'objectif principal, le modèle devra produire deux classes :

- Chat
- Chien

Le modèle sera ensuite entraîné ou ajusté à l'aide de nos photographies étiquetées.

Nous pourrons commencer en gelant les couches d'extraction de caractéristiques et en entraînant uniquement la couche finale.

Si les performances sont insuffisantes, nous pourrons envisager un ajustement partiel du modèle.

### Exigence pédagogique

Il ne suffit pas d'appeler une API externe qui retourne « chat » ou « chien ».

Nous souhaitons comprendre et démontrer :

- Le fonctionnement général d'un CNN.
- Le rôle du modèle pré-entraîné.
- L'apprentissage par transfert.
- La préparation des données.
- La séparation des données d'entraînement et de test.
- Le processus d'entraînement.
- Les prédictions.
- L'évaluation des performances.
- Les limites du modèle.

Codex doit nous expliquer ces concepts pendant le développement.

---

## 7. Jeu de données

### Jeu de données retenu

**Oxford-IIIT Pet Dataset**

Source officielle :

https://www.robots.ox.ac.uk/~vgg/data/pets/

Ce jeu de données comprend :

- 7 349 photographies.
- 37 catégories de races.
- 12 races de chats.
- 25 races de chiens.

Il est particulièrement intéressant puisqu'il permet de développer les deux fonctionnalités avec une seule source de données.

### Préparation des données

Pour la classification binaire, les 37 races seront regroupées en deux catégories :

- Chat
- Chien

Pour la classification avancée, les étiquettes correspondant aux 37 races pourront être conservées.

Les images devront être :

1. Chargées et vérifiées.
2. Redimensionnées selon les exigences du modèle.
3. Normalisées.
4. Associées à leurs bonnes étiquettes.
5. Réparties entre entraînement, validation et test.

### Répartition des données

Répartition envisagée :

- 70 % pour l'entraînement.
- 15 % pour la validation.
- 15 % pour le test.

Cette répartition est une proposition. Nous pouvons également utiliser les partitions officielles du jeu de données lorsqu'elles sont appropriées, en conservant un ensemble de test indépendant.

La répartition devra être reproductible, stratifiée et éviter toute fuite de données.

Les augmentations d'images doivent uniquement être appliquées aux données d'entraînement.

---

## 8. Architecture du système

Le système comportera quatre composantes principales.

### Composante A — Préparation des données

Responsabilités :

- Télécharger ou charger les photographies.
- Lire les étiquettes.
- Préparer les données.
- Séparer les ensembles.
- Appliquer les transformations nécessaires.

### Composante B — Modèle d'intelligence artificielle

Responsabilités :

- Charger MobileNetV3 pré-entraîné.
- Adapter sa couche de classification.
- Entraîner le modèle.
- Sauvegarder les poids du modèle.
- Charger le modèle entraîné.
- Prédire la catégorie d'une nouvelle image.

### Composante C — Classification et organisation des images

Responsabilités :

- Accepter plusieurs images.
- Faire une prédiction pour chacune.
- Enregistrer la catégorie et le score de confiance.
- Organiser les images par catégorie.
- Gérer les fichiers invalides.
- Produire une archive ZIP.

### Composante D — Interface utilisateur

Responsabilités :

- Afficher le nom de l'application IAnimaux.
- Permettre l'importation de photographies.
- Lancer la classification.
- Afficher la progression.
- Présenter les résultats.
- Montrer la catégorie et le score pour chaque photo.
- Afficher des statistiques sommaires.
- Permettre de télécharger les images classées.

### Flux de fonctionnement

```text
Photographies importées
          |
          v
Prétraitement des images
          |
          v
Modèle MobileNetV3
          |
          v
Prédiction Chat / Chien
          |
          v
Score de confiance
          |
          v
Organisation des photographies
          |
          v
Affichage des résultats
          |
          v
Téléchargement ZIP
```

---

## 9. Interface utilisateur

Nous souhaitons une interface graphique simple, moderne et fonctionnelle réalisée avec Streamlit.

### Écran principal

Nom : **IAnimaux**

Sous-titre : *Classification intelligente des photographies d'animaux*

Fonctionnalités :

- Zone permettant d'importer plusieurs images.
- Bouton « Lancer la classification ».
- Barre de progression.
- Nombre d'images analysées.
- Nombre de chats détectés.
- Nombre de chiens détectés.
- Tableau des résultats.
- Prévisualisation des images.
- Bouton « Télécharger les photos classées ».

### Tableau des résultats

Exemple fictif :

| Fichier | Prédiction | Confiance |
|---|---|---|
| photo_001.jpg | Chat | 98,4 % |
| photo_002.jpg | Chien | 96,1 % |
| photo_003.jpg | Chat | 92,7 % |

Les véritables résultats devront provenir du modèle entraîné.

Ne jamais coder en dur des prédictions fictives dans le système final.

### Gestion de l'incertitude

Si le modèle est incertain, le système pourra recommander une vérification manuelle.

Un seuil de confiance pourra être configuré.

Il devra être choisi et justifié à l'aide de l'ensemble de validation, et non arbitrairement présenté comme une garantie de fiabilité.

Les photos difficiles à analyser pourront être dirigées vers `A_verifier/`.

---

## 10. Évaluation du modèle

L'évaluation est une partie essentielle du projet.

Le rapport devra contenir des mesures quantitatives calculées sur un ensemble de test indépendant.

### Métriques principales

**Exactitude (Accuracy)**

Pourcentage d'images correctement classées.

**Précision (Precision)**

Proportion des images prédites dans une catégorie qui appartiennent réellement à cette catégorie.

**Rappel (Recall)**

Proportion des images d'une catégorie correctement retrouvées par le modèle.

**F1-score**

Mesure combinant précision et rappel.

**Matrice de confusion**

Tableau permettant d'identifier les erreurs de classification.

**Temps d'exécution**

Durée nécessaire pour classifier un ensemble de photographies.

### Résultats à produire

- Un tableau des performances.
- Une matrice de confusion.
- Une courbe d'entraînement et de validation, lorsque pertinente.
- Des exemples de bonnes classifications.
- Des exemples d'erreurs de classification.
- Une interprétation des résultats.

Les résultats doivent être obtenus expérimentalement.

Ne jamais inventer de métriques ou de performances.

### Comparaison facultative

Si le temps le permet, comparer :

1. Un modèle classique utilisant des caractéristiques d'images et un classificateur scikit-learn.
2. Le modèle CNN pré-entraîné MobileNetV3.

L'objectif serait d'étudier les différences de performance et les avantages de l'apprentissage par transfert.

Cette comparaison est facultative.

---

## 11. Reconnaissance des races — Extension facultative

La reconnaissance des races doit être développée seulement après la réalisation complète de la classification chats/chiens.

### Approche possible

Adapter une nouvelle version de MobileNetV3 pour classifier les images parmi les 37 races du jeu de données.

Cette tâche est plus difficile que la classification binaire.

### Résultats envisagés

L'application pourrait afficher :

- Type d'animal.
- Race prédite.
- Score de confiance.

Par exemple :

```text
Animal : Chien
Race prédite : Golden Retriever
Confiance : 87 %
```

### Limites

La reconnaissance des races comporte des difficultés :

- Similarités visuelles entre certaines races.
- Qualité variable des photographies.
- Animaux croisés.
- Races absentes du jeu de données.
- Scores de confiance qui ne garantissent pas la justesse de la prédiction.

Les races prédites doivent être présentées comme des estimations visuelles, et non comme une identification génétique certaine.

---

## 12. Organisation du travail individuel

Le projet est réalisé par une seule personne, responsable de l'ensemble des tâches :

- Préparation des données et pipeline d'entraînement.
- Adaptation et entraînement de MobileNetV3.
- Évaluation quantitative et analyse des erreurs.
- Interface Streamlit, prédictions, classement et archive ZIP.
- Tests, documentation, rapport et présentation orale.

Le travail sera séquentiel et progressif. La classification chats/chiens,
l'évaluation et la documentation sont prioritaires sur les extensions.
Chaque composante doit être comprise et pouvoir être expliquée à l'enseignant.

---

## 13. Calendrier du développement

### 18–22 novembre

- Initialisation du projet.
- Installation de l'environnement Python.
- Téléchargement du jeu de données.
- Exploration des données.
- Compréhension du modèle retenu.
- Préparation de la structure du projet.

### 23–27 novembre

- Mise en place de MobileNetV3.
- Entraînement du classificateur chats/chiens.
- Premières mesures de performance.
- Construction initiale de l'interface Streamlit.

### 28 novembre – 2 décembre

- Intégration du modèle dans l'application.
- Importation multiple des images.
- Prédictions.
- Classement automatique.
- Production du ZIP.
- Tests fonctionnels.

### 3–5 décembre

- Évaluation finale.
- Matrice de confusion.
- Graphiques.
- Analyse des limites.
- Rédaction du rapport.
- Reconnaissance des races seulement si le reste est terminé.

### 6–7 décembre

- Corrections finales.
- Vérification du code.
- Vérification des installations.
- Finalisation du README.
- Préparation et répétition de la présentation.

### 8 décembre

Remise du projet.

### 9 décembre

Présentation orale et démonstration.

---

## 14. Exigences du rapport final

Le rapport doit comporter **au maximum 10 pages, références comprises**.

Il doit respecter la structure exigée par l'enseignant.

### Section 1 — Définition de la tâche

Présenter :

- La problématique.
- Les objectifs.
- Les entrées et sorties du système.
- Les fonctionnalités.
- La pertinence de l'intelligence artificielle.

### Section 2 — Brève revue de la littérature

Présenter :

- Les approches existantes de classification d'images.
- Les réseaux de neurones convolutifs.
- L'apprentissage par transfert.
- MobileNetV3.
- Les références utilisées.
- La justification des choix techniques.

### Section 3 — Matériel et méthodes

Présenter :

- Python et les bibliothèques utilisées.
- Le jeu de données Oxford-IIIT Pet.
- La préparation des données.
- L'architecture du modèle.
- La méthode d'entraînement.
- Les principales étapes du système.
- Les critères d'évaluation.

### Section 4 — Résultats et discussion

Présenter :

- Les résultats réels.
- Les métriques calculées.
- La matrice de confusion.
- Les graphiques pertinents.
- Des captures d'écran de l'application.
- Un exemple concret de classification.
- L'interprétation des résultats.
- Les limites du système.

### Section 5 — Conclusion

Présenter :

- Les réalisations.
- Les objectifs atteints.
- Les difficultés rencontrées.
- Les limites.
- Les améliorations possibles.

### Références bibliographiques

Inclure les références du jeu de données, des modèles, des publications scientifiques et des outils utilisés.

---

## 15. Livrables obligatoires

Le projet devra être remis dans une archive ZIP nommée :

`Groupe{k}Final.zip`

Elle devra contenir :

```text
Groupe{k}Final.zip
│
├── Groupe{k}Final.pdf
│
├── Groupe{k}FinalCode/
│   ├── app.py
│   ├── requirements.txt
│   ├── ...
│   └── autres fichiers nécessaires
│
└── README.md
```

La structure interne du code peut être adaptée selon les besoins.

Le README doit expliquer :

- Les prérequis.
- L'installation de Python et des dépendances.
- Le téléchargement éventuel du jeu de données.
- L'entraînement ou le chargement du modèle.
- Le lancement de l'application.
- L'utilisation des fonctionnalités.
- La reproduction des résultats.

Les ressources nécessaires à la démonstration doivent être disponibles et testées.

---

## 16. Critères d'évaluation du professeur

| Critère | Pondération |
|---|---:|
| Définition du problème et pertinence IA | 15 % |
| Revue de la littérature | 10 % |
| Méthodologie et implémentation | 25 % |
| Résultats, évaluation et analyse | 20 % |
| Qualité du rapport | 10 % |
| Présentation orale et démonstration | 20 % |
| Total | 100 % |

Il faut tenir compte de ces critères dans les décisions de développement.

En particulier, le projet doit comporter une méthodologie claire, une évaluation quantitative et une démonstration fonctionnelle.

---

## 17. Intégrité académique

L'énoncé autorise les bibliothèques et modèles pré-entraînés, à condition de citer les sources et de comprendre le fonctionnement de ces outils.

L'utilisation d'outils d'intelligence artificielle générative, comme Codex, doit être déclarée lorsqu'ils contribuent à la production du code ou du rapport.

Nous devons donc :

- Comprendre le code produit.
- Vérifier les résultats.
- Documenter les sources.
- Déclarer l'utilisation de Codex.
- Ne jamais inventer de résultats.
- Être capables d'expliquer les algorithmes et les choix techniques.
- Produire notre propre analyse des résultats.

---

## 18. Instructions à Codex

Tu es notre assistant de développement pour le projet IAnimaux.

Tu dois nous accompagner dans la réalisation d'un système intelligent fonctionnel, pédagogique et conforme à l'énoncé du cours INF1853.

### Principes de développement

1. Privilégier la simplicité et la lisibilité.
2. Utiliser Python comme langage principal.
3. Utiliser PyTorch et Torchvision pour l'IA.
4. Utiliser MobileNetV3 comme modèle de départ.
5. Utiliser Streamlit pour l'interface.
6. Utiliser Oxford-IIIT Pet Dataset pour les données.
7. Expliquer les concepts d'IA importants.
8. Éviter les architectures complexes et le surdéveloppement.
9. Garder les fonctionnalités avancées facultatives.
10. Ne jamais compromettre la livraison de l'objectif principal.

### Qualité du code

- Code modulaire.
- Fonctions simples et explicites.
- Noms de variables compréhensibles.
- Gestion raisonnable des erreurs.
- Dépendances limitées.
- Configuration centralisée lorsque nécessaire.
- Entraînement reproductible.
- Séparation entre entraînement et inférence.
- Tests des principales fonctionnalités.
- Documentation des commandes nécessaires.

### Comportement attendu

Avant toute grande étape, explique brièvement :

- Ce que nous allons développer.
- Pourquoi c'est nécessaire.
- Comment cela fonctionne.
- Comment vérifier que l'étape est terminée.

Ne génère pas inutilement une architecture complexe.

Lorsque plusieurs solutions sont possibles, privilégie celle qui est la plus facile à comprendre, à maintenir et à présenter dans un cours d'introduction à l'intelligence artificielle.

Ne lance pas d'entraînement long ou de téléchargement volumineux sans expliquer les ressources nécessaires.

Ne crée pas de résultats expérimentaux fictifs.

### Priorités

**Priorité 1 :** modèle de classification chats/chiens fonctionnel.

**Priorité 2 :** application Streamlit et classement automatique.

**Priorité 3 :** évaluation rigoureuse et reproductible.

**Priorité 4 :** documentation et préparation de la démonstration.

**Priorité 5 :** reconnaissance des races.

---

## 19. Première tâche à réaliser

Commence par analyser ce document et proposer un plan d'implémentation technique détaillé, mais adapté à une seule personne débutante en intelligence artificielle.

Ensuite :

1. Propose une structure de dossiers simple et propre pour le dépôt Git.
2. Identifie les dépendances Python nécessaires.
3. Explique comment préparer notre environnement de développement.
4. Propose la première étape de développement.
5. Explique comment nous pourrons tester cette première étape.

Ne développe pas immédiatement toutes les fonctionnalités.

Nous voulons construire le projet progressivement, comprendre chaque composante et obtenir un prototype fonctionnel suffisamment tôt pour avoir le temps de réaliser les expériences et le rapport final.

**Objectif final : livrer IAnimaux avant le 8 décembre 2026, avec une démonstration fonctionnelle, un modèle correctement évalué et un rapport académique de qualité.**

## 20. Compléments pédagogiques approuvés

Le projet doit rendre visible la compréhension des techniques du cours :

- Exploration des données, vérification des étiquettes et répartition des classes.
- Illustration du prétraitement et séparation reproductible des ensembles.
- Petits calculs expliqués : convolution, normalisation, softmax, entropie croisée,
  mise à jour par gradient et métriques depuis une matrice de confusion.
- Suivi des pertes et performances d'entraînement et de validation.
- Explication des couches gelées, du taux d'apprentissage et du surapprentissage.
- Comparaison minimale avec une prédiction systématique de la classe majoritaire.
- Analyse de bonnes prédictions et d'erreurs réelles.

Les exemples pédagogiques seront séparés des résultats expérimentaux.
Un carnet pédagogique sera ajouté progressivement ; voir
`docs/demarche_pedagogique.md`. Les calculs détaillés pourront rester dans le
carnet afin de respecter les 10 pages du rapport, références comprises.

Le protocole retenu pour commencer conserve le test officiel et sépare
`trainval` en 80 % d'entraînement et 20 % de validation. Les choix et le seuil
éventuel seront déterminés sur la validation, jamais sur le test final.

L'alignement précis avec les notions vues en cours sera complété lorsque le
plan de cours ou la liste des chapitres sera disponible. Cela n'empêche pas de
préparer l'environnement et les données.

## 21. Séparation et chargement réalisés le 19 novembre 2026

La séparation officielle `trainval` a été divisée en 80 % d'entraînement et 20 %
de validation, avec stratification par race et graine aléatoire 42. Les listes
sont enregistrées dans `results/splits/` :

- Entraînement : 2 944 images (950 chats et 1 994 chiens).
- Validation : 736 images (238 chats et 498 chiens).
- Test : 3 669 images officielles conservées (1 183 chats et 2 486 chiens).

Les 37 races sont présentes dans chaque ensemble. Les empreintes des photos
ont été revérifiées ; aucun identifiant ou doublon exact ne chevauche les ensembles.
La reproduction des CSV à l'identique a été vérifiée.

`config.py` centralise les chemins, la graine et la fraction de validation.
Le chargeur applique un redimensionnement à 256, un recadrage aléatoire 224 × 224
et un retournement horizontal de probabilité 0,5 uniquement à l'entraînement.
La validation et le test conservent le recadrage central et la normalisation
officielle. Les originaux ne sont pas modifiés.

Les 16 tests passent et des lots d'entraînement et de validation ont été chargés
avec succès. Aucun modèle n'est entraîné à cette étape. La prochaine étape est
l'adaptation binaire de MobileNetV3 Small et un premier essai d'entraînement court.

## 22. Premier modèle entraîné le 19 novembre 2026

Le modèle MobileNetV3 Small pré-entraîné sur ImageNet a été adapté à deux sorties.
Le bloc classificateur entier est entraîné (592 898 paramètres), tandis que
l'extracteur (927 008 paramètres) et ses statistiques BatchNorm sont gelés.

L'expérience `first_run` a réalisé trois époques sur CPU avec quatre threads,
des lots de 32, Adam, un taux d'apprentissage de 0,001 et une entropie croisée
non pondérée. Durée entraînement et validation : 404,36 s, environ 6 min 44 s.
La graine est 42. Le test n'a pas été utilisé.

Le critère de sélection fixé est la perte de validation minimale. L'époque 3
est retenue (perte 0,0403), avec 98,23 % d'exactitude et 97,97 % de F1 macro sur
les 736 images de validation. La référence majoritaire atteint 67,66 % sur cette
validation. Ces mesures ne sont pas des résultats de test final.

Le checkpoint est dans `models/first_run/best_model.pt` et son rechargement a
été vérifié. Paramètres, historique, prédictions, métriques et figures sont
dans `results/training/first_run/`. Les 19 tests passent. Le script
`scripts/explain_learning.py` produit séparément des calculs pédagogiques de
convolution, normalisation, softmax, entropie croisée, gradient et métriques.

La prochaine étape est l'intégration de la prédiction dans l'application,
avant le classement ZIP et l'évaluation finale. Aucun seuil de confiance n'est
encore sélectionné. La priorité reste chats/chiens.

## 23. Interface d'importation et inférence réalisées le 19 novembre 2026

Streamlit permet maintenant d'importer plusieurs photos JPG/JPEG, PNG ou WebP
(20 Mo maximum par photo) et de lancer la classification avec le checkpoint
`models/first_run/best_model.pt`. L'interface affiche progression, tableau,
catégories, scores, compteurs et aperçus. Le modèle reste en mode évaluation,
sans suivi de gradients et sans nouvel entraînement.

`ianimaux/predict.py` réutilise le prétraitement de validation, vérifie chaque
photo et isole les erreurs de fichiers. Les originaux ne sont pas modifiés.
Les résultats sont effacés si la sélection change ; les noms répétés restent
des entrées distinctes. Aucun seuil arbitraire de confiance n'est appliqué.

Les 27 tests passent, dont le parcours Streamlit avec le vrai checkpoint. Les
probabilités d'inférence correspondent aux prédictions sauvegardées sur deux
photos de validation. Aucune photo du test indépendant n'a été utilisée.
Voir `docs/utilisation_application.md` et `results/interface/verification_inference.json`.

La prochaine étape est l'organisation des copies dans Chats/ et Chiens/ et la
production du ZIP, avec gestion des noms en double et des fichiers invalides.

## 24. Corrections humaines ajoutées le 19 novembre 2026

Après un essai utilisateur de 19 photos, une erreur a été signalée sur
`vraies photos/images (15).jpeg`, un chat sphynx. L'erreur est reproduite :
Chien 54,07 %, Chat 45,93 %. Le recadrage central retire les yeux et les oreilles,
un facteur plausible à examiner, sans cause unique démontrée. Le jeu d'entraînement
contient déjà 80 sphynx et la validation 20. L'essai utilisateur est exploratoire,
pas une nouvelle évaluation indépendante annotée.

L'application permet maintenant de choisir une photo et sa catégorie correcte,
avec commentaire facultatif. Les corrections et photos explicitement validées
sont conservées localement dans une base SQLite, avec historique et empreinte
du modèle. La même photo réutilise sa correction, tandis que le tableau conserve
les sorties IA brutes séparément de la catégorie humaine retenue.

La correction Chat du sphynx est enregistrée d'après l'indication utilisateur.
Cette mémorisation ne modifie pas les poids et ne garantit pas une amélioration
sur d'autres photos. La priorité reste la classification Chat/Chien ; un commentaire
« Sphynx » n'est pas une reconnaissance de race.

`scripts/export_feedback.py` prépare un instantané d'exemples externes pour un
futur entraînement. Les photos Oxford connues, notamment validation et test,
sont exclues de cet export par empreintes exactes de fichiers ou pixels.
L'export `data/feedback/export_01` contient un exemple corrigé ; aucun réentraînement
n'a été lancé et `scripts/train.py` ne le consomme pas automatiquement.

Les 34 tests passent, dont un parcours de correction entre deux sessions et un
contrôle des poids et scores IA inchangés. Voir `docs/corrections_humaines.md`.
Le classement ZIP reste la prochaine fonctionnalité principale.

## 25. Généralisation et modèles comparés le 19 novembre 2026

Le besoin métier a été précisé : le refuge traite surtout de nouvelles photos.
La réapplication automatique d'une ancienne correction au même fichier est
retirée. Une correction ajuste le lot courant et reste une annotation pour
le réentraînement. Une nouvelle classification utilise les sorties du modèle.
Cette politique remplace la mémorisation automatique décrite à la section 24.

Deux candidats ont été entraînés explicitement depuis `first_run`, avec
2 944 photos Oxford et la correction externe du sphynx, trois époques, Adam,
un taux de 0,0001, des lots de 32 et l'extracteur/BatchNorm gelés. Le test officiel
reste inutilisé. Les critères d'activation ont été fixés avant les résultats :
F1 macro et exactitude globale non inférieurs à la référence, et rappel
strictement supérieur sur les 20 sphynx de validation.

| Modèle | Exactitude validation | F1 macro | Sphynx reconnus | Décision |
|---|---:|---:|---:|---|
| first_run | 98,23 % | 97,97 % | 17/20 | Référence conservée |
| full_frame_feedback_v1 | 97,28 % | 96,92 % | 19/20 | Non activé : régression globale |
| refined_feedback_v1 | 99,05 % | 98,91 % | 18/20 | Activé |

Le candidat avec marges conserve toute l'image mais réduit la performance globale.
Le candidat retenu conserve le prétraitement initial et poursuit l'apprentissage
avec les données mélangées. Le checkpoint d'époque 3 est sélectionné sur sa
perte minimale de validation (0,0283). L'empreinte du modèle original est inchangée.

Le modèle actif est `models/refined_feedback_v1/best_model.pt`, défini dans
`config.py`. L'activation est tracée dans `results/model_activation.json` et les
comparaisons dans `results/comparisons/`. La photo corrigée est maintenant prédite
Chat à 65,43 % par le réseau, origine Modèle, sans correction automatique. Comme
elle a servi à entraîner, elle n'est pas une preuve indépendante de généralisation.

Les 40 tests passent. Deux erreurs subsistent parmi les sphynx de validation ;
aucune garantie sur toutes les futures photos n'est revendiquée. La poursuite
d'entraînement et l'ajout de données sont simultanés : le gain ne peut pas être
attribué à la seule correction. Voir `docs/reentrainement_generalisation.md`.
Le classement ZIP reste la prochaine fonctionnalité principale.

## 26. Apparence ludique pastel choisie le 19 novembre 2026

La direction visuelle demandée est ludique, avec couleurs pastel et ambiance
animalière. L'interface utilise lavande, pêche et menthe, une illustration SVG
locale de chat et chien, un parcours en trois étapes, une carte d'importation,
quatre cartes de compteurs et des aperçus de photos encadrés.

La présentation est répartie entre `.streamlit/config.toml`, `assets/styles.css`,
`assets/companions.svg` et `ianimaux/ui.py`. Les polices sont locales ou système.
Le modèle actif et les méthodes de classification ne sont pas modifiés par
cette évolution graphique. Les 40 tests passent. Un contrôle dans Chrome a
vérifié l'importation et la classification sur ordinateur et le rendu à une
largeur mobile de 390 pixels, sans débordement horizontal de la page.
Les captures et `verification_design.json` sont dans `results/interface/`.


## 27. Téléchargement ZIP ajouté le 19 novembre 2026

Le bouton « Télécharger les photos classées (ZIP) » génère au clic
`IAnimaux_photos_classees.zip`, contenant exactement deux dossiers racine :
`chat/` et `chien/`. Ces noms en minuscules et au singulier remplacent les
noms de sortie précédemment envisagés, selon la demande utilisateur.

La catégorie retenue du lot détermine le dossier : correction explicite du lot
si présente, sinon prédiction du modèle. Les anciennes annotations ne sont pas
appliquées à une nouvelle classification. Les octets originaux sont préservés,
les noms répétés sont suffixés pour éviter les écrasements et les chemins des
noms importés sont neutralisés. Les deux dossiers sont présents même si l'un
est vide. Les fichiers invalides sont exclus et signalés ; le bouton est
désactivé si aucune photo n'est valide.

`ianimaux/export.py` utilise la bibliothèque standard zipfile. L'archive est
créée en mémoire par la fonction différée du bouton Streamlit, sans copies
sur le disque du serveur et sans réentraînement. Les 47 tests passent.
Un téléchargement réel a été réalisé dans Chrome : les deux dossiers et les
photos attendues sont présents, l'intégrité CRC est valide et les octets
correspondent aux originaux. Voir `results/interface/verification_telechargement_zip.json`.

Le parcours importation → classification → revue → téléchargement est fonctionnel.
Les prochaines priorités sont l'évaluation finale indépendante, l'analyse des
limites et la préparation des livrables académiques ; les races restent facultatives.

## 28. Version binaire fixée avant le test — 21 novembre 2026

L'extension de races a été retirée par l'utilisatrice. La priorité reste la
classification binaire. À sa demande, le modèle actif refined_feedback_v1 est
conservé sous results/evaluation/binary_final_v1/best_model.pt, identique par
SHA-256 d0b66d9beb3e739a81691af6f4364e3a0efd988393ef9cb9bf35282aeaf72520.
Le manifeste fixe l'époque 3, les paramètres d'entraînement et de prétraitement,
les versions logicielles, le protocole de décision et les empreintes des sources
et partitions. Des copies des sources pertinentes et configurations sont conservées.
La commande scripts/freeze_evaluation.py --verify détecte les divergences ;
la création refuse d'écraser un gel existant. Les trois tests de contrôle couvrent
l'altération des poids conservés, des sources et des partitions.
Aucune image de test ouverte, aucune inférence ni métrique finale produite.
Prochaine étape : évaluer la copie gelée sur les 3 669 images de test, sans
correction humaine ni réglage sur le test. Référence majoritaire fixée à Chien
selon l'entraînement. Voir docs/gel_version_evaluation.md.


## 29. Évaluation finale réalisée — 21 novembre 2026

La copie gelée binary_final_v1 a été évaluée sur les 3 669 images officielles
de test, après contrôle des empreintes des poids, sources, partitions et images.
Aucun réentraînement, seuil ajusté sur le test ou remplacement par correction
humaine. Décision argmax, CPU 4 threads, lots de 32, aucune augmentation.
Résultats : exactitude 98,42 %, F1 macro 98,18 %,
3611 bonnes classifications et 58 erreurs. Matrice (lignes réelles
Chat/Chien, colonnes prédites Chat/Chien) : [[1139, 44], [14, 2472]].
Référence majoritaire Chien fixée sur train : exactitude 67,76 %.
Les sorties, scores par image, erreurs, graphiques et empreintes sont dans
results/evaluation/binary_final_v1/test_results/. Le script
scripts/evaluate_final.py refuse de les écraser. 55 tests passent.
Le manifeste du gel n'est pas modifié : il décrit l'état AVANT cette évaluation.
Le test est désormais consulté ; ne plus le traiter comme inédit pour choisir
le modèle. Les mesures Oxford ne garantissent pas la performance dans un refuge.
Voir docs/evaluation_finale.md. Prochaines priorités : compréhension des erreurs,
rapport final de 10 pages, présentation orale et archive Moodle.


## 30. Rédaction du rapport final — 21 novembre 2026

Les coauteures sont désormais Megan Daoust et Luzia Coelho Bento Muaquesse,
selon la confirmation explicite de l’utilisatrice. Les contributions sont
déclarées équilibrées ; cette précision remplace le périmètre individuel initial.
Le rapport final est rédigé dans docs/rapport_final/ : PDF de dix pages au total,
HTML modifiable, source contenu.json et script de génération. Il adapte la
présentation du gabarit fourni, en suivant les cinq sections de l’énoncé.
Les chiffres proviennent des artefacts locaux, avec références vérifiées et
déclaration explicite des contributions de Codex au code et à la rédaction.
Le numéro de groupe reste à compléter ; le nom Groupe{k}Final.pdf ne peut
être fixé sans ce numéro. Les coauteures doivent relire et valider le rapport.
Aucune soumission ni publication effectuée.
