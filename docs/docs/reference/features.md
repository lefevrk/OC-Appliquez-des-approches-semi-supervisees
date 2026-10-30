# Module `brain_mri.features`

Prétraitement des images et extraction de caractéristiques avec un ResNet18 pré-entraîné. Le détail du flux est décrit dans la page [Extraction des features](../pipeline/features.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `MODEL_NAME` | `"resnet18"` | Nom du modèle, utilisé dans le nom des fichiers produits |
| `EXTRACTION_LAYERS` | `("layer3", "layer4")` | Couches du ResNet18 dont on extrait les features |
| `IMAGE_SIZE` | `224` | Côté des images en entrée du modèle, en pixels |
| `BATCH_SIZE` | `32` | Nombre d'images par lot |
| `IMAGENET_MEAN`, `IMAGENET_STD` | statistiques d'ImageNet | Normalisation, reprise des transformations officielles des poids |

## Prétraitement

::: brain_mri.features.preprocess_equalized

::: brain_mri.features.preprocess_gray

::: brain_mri.features.preprocess_rgb

::: brain_mri.features.ManifestImages

## Extraction

::: brain_mri.features.select_device

::: brain_mri.features.load_extractor

::: brain_mri.features.extract_features

## Sauvegarde

::: brain_mri.features.save_features

::: brain_mri.features.load_features

## Ligne de commande

::: brain_mri.features.main
