# Module `brain_mri.dataset`

Chargement, déduplication et manifeste des données d'IRM. Le flux d'ensemble est décrit dans la page [Préparation](../pipeline/preparation.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `IMAGE_PATTERN` | `"*.jpg"` | Motif des fichiers image recherchés |
| `DATA_SOURCES` | `sans_label`, `cancer`, `normal` | Sous-dossier de `data/raw/` et label de chaque sous-ensemble |
| `INVENTORY_COLUMNS` | 6 colonnes | Colonnes de la table produite par `build_inventory` |
| `REMOVED_COLUMNS` | 7 colonnes | Colonnes du rapport des suppressions |
| `EXCLUDED_IMAGES` | 1 image | Images exclues à la main, car inexploitables, avec la raison |
| `THUMBNAIL_SIZE` | `64` | Côté des miniatures utilisées pour les quasi-doublons, en pixels |
| `NEAR_DUPLICATE_THRESHOLD` | `0.02` | Écart quadratique moyen (RMS) maximal entre miniatures pour un quasi-doublon, pixels dans [0, 1] |
| `ORIENTATIONS` | 8 symétries du carré | Orientations essayées pour chaque image : rotations de 0°, 90°, 180°, 270°, avec ou sans miroir |
| `SPLIT_RANDOM_STATE` | `42` | Graine du découpage |
| `SPLIT_NAMES` | `train`, `validation`, `test` | Valeurs possibles de la colonne `split` |

## Inventaire

::: brain_mri.dataset.build_inventory

## Empreintes

::: brain_mri.dataset.file_hash

::: brain_mri.dataset.add_file_hash

Les tailles du test (`TEST_SIZE`) et de la validation (`VALIDATION_SIZE`) sont dans [`brain_mri.config`](config.md).

## Déduplication

::: brain_mri.dataset.deduplicate_exact

::: brain_mri.dataset.load_thumbnail

::: brain_mri.dataset.NearDuplicateSearch

::: brain_mri.dataset.nearest_neighbors

::: brain_mri.dataset.find_near_duplicate_pairs

::: brain_mri.dataset.deduplicate_near

::: brain_mri.dataset.DeduplicationResult

::: brain_mri.dataset.deduplicate

## Manifeste

::: brain_mri.dataset.write_manifest

::: brain_mri.dataset.read_manifest

## Découpage

::: brain_mri.dataset.split_dataset

::: brain_mri.dataset.read_split

## Ligne de commande

::: brain_mri.dataset.main
