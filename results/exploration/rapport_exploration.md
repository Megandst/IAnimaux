# IAnimaux — exploration réelle d'Oxford-IIIT Pet

Commande de reproduction : `python scripts/prepare_data.py` (ajouter `--download` à la première exécution).

- Images annotées : 7349.
- Images lisibles : 7349 ; illisibles ou absentes : 0.
- Races : 37.
- Effectifs par espèce : {'Chien': 4978, 'Chat': 2371}.
- Partitions officielles : {'trainval': 3680, 'test': 3669}.
- Groupes de doublons exacts de pixels : 0.
- Groupes traversant les partitions officielles : 0.

## Méthode

Les espèces sont lues dans la troisième colonne des annotations officielles :
1 = chat, 2 = chien. La cible du modèle est 0 = Chat, 1 = Chien.
Les noms de fichiers servent à identifier les races, pas à déduire l'espèce.
L'unicité des identifiants et la cohérence des étiquettes sont contrôlées.
Chaque image est vérifiée puis entièrement décodée ; aucune image originale n'est modifiée.
Deux empreintes SHA-256 sont enregistrées : fichier et pixels RGB après orientation EXIF.
La détection de doublons couvre les pixels identiques, pas les photos presque identiques.

## Prétraitement illustré

Le prétraitement officiel des poids MobileNetV3 Small IMAGENET1K_V1 est utilisé :
redimensionnement du petit côté à 256, recadrage central 224 × 224, valeurs divisées
par 255, puis normalisation de chaque canal : `z = (pixel / 255 - moyenne) / écart_type`.
Moyennes RGB : [0.485, 0.456, 0.406] ; écarts-types : [0.229, 0.224, 0.225].
Les calculs sur des pixels réels sont dans `exemples_normalisation.csv`.
Les figures utilisent uniquement des exemples de `trainval` et annulent la
normalisation pour afficher une image interprétable. Le recadrage peut retirer
une partie de l'animal : c'est une limite à examiner.
Aucun poids pré-entraîné n'est téléchargé pour ces transformations.

## Interprétation et prochaine étape

Le nombre de races par espèce contribue au déséquilibre des classes : il faut
rapporter les métriques par classe et le F1 macro, pas seulement l'exactitude.
Les effectifs ci-dessus sont ceux des annotations, avant toute exclusion.
Consulter `images_invalides.csv` et `doublons_exacts.csv` avant de figer les partitions.
Une image illisible ne sera pas silencieusement utilisée pour l'entraînement.
Les éventuels doublons ne devront pas traverser entraînement, validation et test.
La politique d'exclusion sera documentée lors de la création des partitions.
Le test officiel reste réservé à l'évaluation finale ; aucune prédiction ni
sélection de modèle n'est effectuée ici. Les partitions finales ne sont pas encore créées.

Source : [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Les images restent attribuées à leurs propriétaires d'origine ; voir la licence
indiquée sur la page officielle avant toute redistribution.
