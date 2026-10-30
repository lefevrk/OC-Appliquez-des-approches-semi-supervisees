# Module `brain_mri.config`

Chemins, paramètres communs et outils de reproductibilité, centralisés pour que les notebooks et les modules les partagent.

## Chemins

| Nom | Valeur |
|---|---|
| `PROJ_ROOT` | Racine du projet |
| `RAW_DATA_DIR` | `data/raw/` (immuable) |
| `INTERIM_DATA_DIR` | `data/interim/` (manifeste) |
| `PROCESSED_DATA_DIR` | `data/processed/` (découpage, features, labels faibles) |
| `EXTERNAL_DATA_DIR` | `data/external/` |
| `MODELS_DIR` | `models/` |
| `REPORTS_DIR` | `reports/` (rapport des suppressions) |
| `FIGURES_DIR` | `reports/figures/` (figures des notebooks, publiées dans cette documentation) |

## Paramètres communs

| Nom | Valeur | Rôle |
|---|---|---|
| `SEED` | `42` | Graine de tous les algorithmes aléatoires |
| `IMAGE_SIZE` | `224` | Côté des images en entrée du modèle, en pixels |
| `BATCH_SIZE` | `32` | Nombre d'images par lot |
| `TEST_SIZE` | `15` | Nombre d'images annotées réservées au test |
| `VALIDATION_SIZE` | `14` | Nombre d'images annotées réservées à la validation |
| `N_COMPONENTS` | `50` | Composantes conservées par la PCA |
| `N_CLUSTERS` | `2` | Nombre de clusters de K-Means, comme de classes |

## Sorties sans chemin personnel

Les sorties de notebook ne doivent révéler aucun chemin de la machine de l'auteur. `mask_path` remplace la racine du projet par `<PROJECT_ROOT>` et le dossier personnel par `~` (constantes `SAFE_ROOT_LABEL` et `SAFE_HOME_LABEL`). Les fonctions de [suivi des expériences](tracking.md) l'appliquent aux messages de MLflow.

::: brain_mri.config.mask_path

## Reproductibilité

::: brain_mri.config.set_seed
