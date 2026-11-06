# Module `brain_mri.modeling.train`

Entraînement d'un ResNet50 dont seules `layer4` et la couche de classification s'ajustent, et mesure de la détection du `cancer`. Le détail du protocole est décrit dans la page [Méthode semi-supervisée](../pipeline/semi-supervise.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `LEARNING_RATE` | `1e-4` | Taux d'apprentissage d'Adam |
| `TRAIN_BATCH_SIZE` | `16` | Nombre d'images par lot |
| `WEAK_EPOCHS` | `10` | Époques de la phase sur les labels faibles |
| `MAX_EPOCHS` | `30` | Époques au plus de la phase sur les labels forts |
| `PATIENCE` | `5` | Époques sans baisse de la perte de validation avant l'arrêt |
| `DECISION_THRESHOLD` | `0.5` | Probabilité de `cancer` à partir de laquelle une image est classée `cancer` |
| `NUM_CLASSES` | `2` | `normal` et `cancer` |
| `AUGMENT` | retournement, rotation, translation, zoom | Augmentation légère de l'entraînement |

## Modèle

::: brain_mri.modeling.train.create_model

::: brain_mri.modeling.train.count_parameters

## Images et lots

::: brain_mri.modeling.train.LabeledImages

::: brain_mri.modeling.train.make_loader

## Entraînement et mesures

::: brain_mri.modeling.train.fit

::: brain_mri.modeling.train.predict

::: brain_mri.modeling.train.classification_scores
