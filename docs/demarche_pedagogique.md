# Démarche pédagogique

IAnimaux doit démontrer une compréhension de la démarche d'apprentissage
supervisé, en plus de fournir une application fonctionnelle.

## Démonstrations à construire progressivement

1. **Exploration et préparation** : effectifs par espèce et race, dimensions,
   images illisibles, vérification des étiquettes et illustrations avant/après
   prétraitement.
2. **Calculs illustrés** : convolution sur une matrice, normalisation d'un pixel,
   softmax, entropie croisée, petite mise à jour par gradient et calcul des
   métriques depuis une matrice de confusion. Prévoir un carnet
   `notebooks/01_comprendre_ia.ipynb` lorsque ces notions seront abordées.
3. **Entraînement observable** : couches gelées et entraînables, lots, taux
   d'apprentissage, perte et performances par époque, analyse du surapprentissage.
4. **Évaluation** : exactitude, précision, rappel, F1 par classe, F1 macro,
   matrice de confusion, temps de traitement et analyse d'erreurs réelles.
5. **Référence minimale** : comparaison avec une prédiction systématique de la
   classe majoritaire, déterminée sur les données d'entraînement.

Les calculs illustratifs doivent être explicitement identifiés. Ils ne sont pas
des résultats expérimentaux. Les mesures finales proviendront du modèle entraîné.
Le score softmax n'est pas une garantie de justesse ni un détecteur de photos
sans chat ou chien.

## Protocole prévu

Conserver le test officiel Oxford-IIIT Pet. Diviser `trainval` en 80 %
d'entraînement et 20 % de validation, de manière stratifiée et reproductible.
Vérifier les annotations et les doublons avant de figer les partitions.
Enregistrer les listes de fichiers et la graine aléatoire. Les augmentations
concernent uniquement l'entraînement.

Utiliser la validation pour choisir les paramètres, le meilleur modèle et un
éventuel seuil de confiance. Réserver le test à l'évaluation finale. Si un seuil
est appliqué, rapporter aussi la proportion d'images acceptées automatiquement.

## Étape 1 : comprendre l'environnement

- Un environnement virtuel isole les dépendances du projet.
- Une vérification d'import détecte une bibliothèque absente ou inutilisable.
- Un calcul sur un tenseur vérifie que PyTorch peut réellement s'exécuter.
- Un passage dans MobileNetV3 avec `weights=None` contrôle l'architecture sans
  télécharger de poids et sans produire de prédiction utile.
- Le CPU suffit au premier contrôle ; MPS est une accélération facultative sur
  les Mac compatibles.

À pouvoir expliquer : quelle différence y a-t-il entre installer une bibliothèque,
construire un réseau, charger des poids pré-entraînés et entraîner un modèle ?

## Périmètre et échéance

Projet réalisé par une seule personne. Prototype chats/chiens complet visé au
1er décembre 2026, évaluation du 2 au 4 décembre, finalisation du 5 au 7 décembre.
Remise le 8 décembre et présentation le 9 décembre. Les races et une comparaison
avec un autre algorithme restent facultatives. Commencer les notes du rapport
dès la préparation des données.

## Étape 2 : exploration des données

Voir `preparation_donnees.md` pour la lecture des annotations, la vérification des
images, la recherche de doublons et le calcul de normalisation d'un pixel.
Le script `scripts/prepare_data.py` produit les résultats réels et les figures
dans `results/exploration/`. Il ne crée pas encore les partitions finales et
n'entraîne aucun modèle.

## Étape 3 : séparation et chargement réalisés

Les listes finales sont enregistrées dans `results/splits/` : 2 944 images pour
l'entraînement, 736 pour la validation et les 3 669 images du test officiel.
La graine est fixée à 42 et la séparation est stratifiée par race. Les 37 races
sont présentes dans les trois ensembles. La reproductibilité des CSV a été
vérifiée par une deuxième exécution sur les données réelles.

Les augmentations (recadrage aléatoire et retournement horizontal) sont réservées
à l'entraînement. Le chargement par lots et la répétabilité du prétraitement de
validation ont été contrôlés. Voir `separation_donnees.md` pour comprendre la
stratification, la graine et les dimensions d'un lot d'images.

## Étape 4 : entraînement et calculs

La boucle d'entraînement et le modèle binaire sont implémentés. L'extracteur
pré-entraîné et ses statistiques BatchNorm restent gelés ; seul le bloc
classificateur apprend. Voir `entrainement_modele.md` pour comprendre les logits,
le softmax, l'entropie croisée, la rétropropagation et le choix du checkpoint.

`scripts/explain_learning.py` exécute les exemples de convolution, normalisation,
perte, gradient et métriques. Ses valeurs artificielles sont identifiées et
enregistrées séparément des résultats réels de l'entraînement. Le test indépendant
n'est pas utilisé pour sélectionner le modèle.

## Retour humain et nouvelles données

Le cas du sphynx permet d'expliquer l'analyse d'une erreur réelle, l'effet du
recadrage, la création d'une étiquette humaine et la différence entre mémoire
d'une correction et généralisation par apprentissage. Les sorties IA brutes
restent visibles pour ne pas transformer une correction humaine en succès
automatique fictif. Voir `corrections_humaines.md` pour le stockage, l'export et
la protection des partitions avant un futur réentraînement.


## Évaluation finale réalisée

La copie gelée a été évaluée sur le test officiel sans ajustement du modèle.
Voir [evaluation_finale.md](evaluation_finale.md) : les calculs de précision,
rappel et F1 y sont expliqués à partir de la vraie matrice de confusion.
Le test est désormais consulté. La suite porte sur l'interprétation, le rapport
final et la démonstration, en conservant les résultats obtenus.
