# Version fixée avant l’évaluation finale

Le 21 novembre 2026, la version binaire active a été conservée sous l'identifiant
**binary_final_v1**. L'estimation des races retirée ne fait pas partie de cette
version. Cette étape ne calcule aucune performance sur le test indépendant.

## Modèle et paramètres retenus

- Source : `models/refined_feedback_v1/best_model.pt`.
- Copie destinée à l'évaluation : `results/evaluation/binary_final_v1/best_model.pt`.
- SHA-256 des deux fichiers :
  `d0b66d9beb3e739a81691af6f4364e3a0efd988393ef9cb9bf35282aeaf72520`.
- Architecture : MobileNetV3 Small, deux sorties dans l'ordre Chat (0), Chien (1).
- Extracteur convolutif et BatchNorm gelés ; classificateur entraîné par transfert.
- 1 519 906 paramètres au total, dont 592 898 entraînables lors de l'apprentissage.
- Poursuite depuis `first_run`, 3 époques, Adam, taux 0,0001, lots de 32, graine 42.
- Données d'apprentissage : 2 944 images Oxford et 1 correction externe validée.
- Sélection : époque 3, perte de validation minimale ; 736 images de validation.
- Validation connue : exactitude 99,05 %, F1 macro 98,91 %, perte 0,02832.

Ces mesures sont des résultats de validation, pas des résultats finaux de test.
Le manifeste conserve aussi l'empreinte du modèle initial et celle du manifeste
des corrections utilisées, présentes dans la configuration d'entraînement.

## Prétraitement et décision fixés

Orientation EXIF, conversion RGB, redimensionnement du petit côté à 256 pixels,
interpolation bilinéaire avec anticrénelage, recadrage central 224 × 224, conversion
en valeurs entre 0 et 1 puis normalisation ImageNet : moyennes
[0,485 ; 0,456 ; 0,406], écarts-types [0,229 ; 0,224 ; 0,225].
Aucune augmentation pendant l'évaluation. Le modèle sera en mode `eval`, avec
`torch.inference_mode`, sur CPU (4 threads), lots de 32 et sans mélange des images.
La décision est l'argmax des deux logits ; une égalité retient l'indice 0.
Le score affiché est le softmax. Aucun seuil de rejet, aucune correction humaine
et aucun remplacement par une annotation historique ne seront appliqués.

## Éléments conservés et contrôles

`results/evaluation/binary_final_v1/manifest.json` enregistre les paramètres,
les versions de Python et des bibliothèques, le commit Git présent au moment
du gel, les empreintes des partitions et celles du code pertinent. Les sources
et configurations sont aussi copiées sous `snapshot/`. Le commit seul ne suffit
pas à identifier un travail non commité : les copies et empreintes font référence.
Les CSV de partitions ont été lus uniquement pour contrôler leurs empreintes ;
aucune image de test n'a été chargée et aucune prédiction n'a été produite.

Pour contrôler le gel avant l'évaluation :

```bash
python scripts/freeze_evaluation.py --verify
```

Le contrôle échoue si les poids actifs, la copie conservée, les sources suivies,
les partitions ou les versions logicielles suivies ont changé. La création
initiale utilise la même commande sans `--verify` et refuse d'écraser un gel
existant. Il s'agit d'une trace vérifiable, pas d'une protection contre la
modification volontaire simultanée du manifeste et des fichiers.

Le modèle et le prétraitement ne doivent plus être ajustés en fonction du test.
En cas de changement nécessaire avant son ouverture, documenter la raison et
conserver cet état de référence. Après l'évaluation, toute modification constitue
une nouvelle expérience et ne doit pas être présentée comme validée sur un test
resté indépendant.

## Prochaine étape

Évaluer cette copie sur les 3 669 images officielles de test : exactitude,
précision, rappel, F1 par classe, F1 macro et matrice de confusion. La référence
majoritaire est fixée à **Chien**, selon l'entraînement. Aucune sélection de modèle
ni de seuil ne sera faite sur le test. Cette exécution reste à réaliser.


## Mise à jour après le gel

L'évaluation prévue a ensuite été exécutée le 21 novembre 2026. Le manifeste
et la copie gelée restent inchangés. Voir les [résultats finaux](evaluation_finale.md).
