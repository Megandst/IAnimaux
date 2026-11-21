# Étape 2 — comprendre les données d'IAnimaux

Cette fiche présente l'exploration réalisée à l'étape 2. La séparation et le
chargement sont maintenant réalisés et décrits dans [l'étape 3](separation_donnees.md).

## Ce que nous faisons et pourquoi

Un apprentissage supervisé nécessite des entrées (photos) et des réponses connues
(étiquettes). Des étiquettes incohérentes, des fichiers illisibles ou des copies
réparties entre entraînement et test peuvent fausser l'apprentissage ou l'évaluation.
Nous contrôlons donc les données avant de choisir ou d'entraîner un modèle.

Le code de lecture est dans `ianimaux/data.py` ; le script reproductible est
`scripts/prepare_data.py`. Les résultats obtenus sont dans
`results/exploration/rapport_exploration.md`. Les tests utilisent de petites images
artificielles pour vérifier la gestion des anomalies ; ils ne mesurent pas l'IA.

## Lire une annotation

Chaque ligne des fichiers officiels `trainval.txt` et `test.txt` contient :

```text
identifiant_image    identifiant_classe    espece    identifiant_race_dans_espece
```

L'espèce est codée 1 pour chat et 2 pour chien. Notre cible utilisera 0 pour Chat
et 1 pour Chien. Nous conservons également la classe de race pour vérifier les
correspondances et, plus tard, stratifier la séparation par race. Cela ne signifie
pas que nous entraînons un modèle de reconnaissance des races.

## Observer le déséquilibre

La proportion d'une classe est son effectif divisé par le nombre total d'images.
Pour une référence qui prédit systématiquement la classe majoritaire, son
exactitude sur un ensemble est simplement la proportion de cette classe dans
cet ensemble. La classe à prédire sera déterminée sur l'entraînement uniquement.
Un bon résultat devra dépasser cette référence et préserver des performances
utiles pour les chats comme pour les chiens.

## Transformer les images

Les images n'ont pas toutes les mêmes dimensions ni le même mode de couleur.
Nous appliquons l'orientation EXIF puis une conversion RGB pour fournir trois
canaux de couleur au réseau. Le prétraitement associé aux poids MobileNetV3
Small redimensionne le petit côté à 256 et effectue un recadrage central 224 × 224.
Le tenseur obtenu a la forme `(3, 224, 224)` : canaux, hauteur, largeur.

La conversion et la normalisation sont deux opérations distinctes :

```text
x = valeur_du_pixel / 255
z = (x - moyenne_du_canal) / ecart_type_du_canal
```

Exemple pédagogique, sans lien avec une mesure de performance : pour une valeur
rouge de 128, `x = 128 / 255 ≈ 0,50196`, puis
`z = (0,50196 - 0,485) / 0,229 ≈ 0,07406`.
Une valeur normalisée peut être négative : ce n'est pas une erreur.
Les constantes proviennent du prétraitement des poids pré-entraînés ; elles ne
sont pas estimées à partir de notre test.

Le fichier `exemples_normalisation.csv` montre ce calcul sur de vrais pixels
après redimensionnement et recadrage. Pour afficher les images transformées,
nous annulons la normalisation. Montrer directement le tenseur normalisé comme
une photo produirait des couleurs trompeuses.

Le recadrage peut supprimer une partie du sujet. Nous utilisons ici le
prétraitement de validation/inférence ; les augmentations aléatoires, réservées
à l'entraînement, seront ajoutées lors de la création des partitions.

## Éviter les fuites de données

Une copie exacte d'une image ne doit pas être utilisée à la fois pour apprendre
et pour évaluer le modèle. Nous calculons des empreintes SHA-256 sur les fichiers
et sur leurs pixels RGB orientés pour repérer des copies, même si leur compression
ou leurs métadonnées diffèrent sans changer les pixels décodés.
Cela ne détecte pas une image recadrée, légèrement retouchée ou recompressée avec
perte. L'absence de doublons exacts ne prouve donc pas l'absence de toute fuite.

À la fin de l'exploration, les partitions finales n'étaient pas encore créées.
L'étape 3 a conservé le test officiel et séparé `trainval` de manière stratifiée
et reproductible. Aucune exclusion n'a été nécessaire sur les données auditées.
Les illustrations utilisent uniquement `trainval` ; le test ne sert pas à
sélectionner des exemples ou à améliorer le modèle.

## Ce que je dois pouvoir expliquer au professeur

- Quelle différence y a-t-il entre une entrée et une étiquette ?
- Pourquoi compter les classes avant l'entraînement ?
- Pourquoi convertir en RGB, redimensionner et normaliser ?
- Comment calculer la valeur normalisée d'un pixel ?
- Comment un doublon peut-il rendre une évaluation trompeuse ?
- Pourquoi distinguer entraînement, validation et test ?

## Sources

- Parkhi et al., *Cats and Dogs*, CVPR 2012 ;
  [jeu Oxford-IIIT Pet et annotations](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- [Torchvision : OxfordIIITPet](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html).
- [Torchvision : MobileNetV3 Small et prétraitement des poids](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
