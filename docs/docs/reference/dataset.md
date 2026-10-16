# Module `brain_mri.dataset`

Chargement, déduplication et manifeste des données d'IRM. Le flux d'ensemble est décrit dans la page [Préparation](../pipeline/preparation.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `IMAGE_PATTERN` | `"*.jpg"` | Motif des fichiers image recherchés |
| `DATA_SOURCES` | `sans_label`, `cancer`, `normal` | Sous-dossier de `data/raw/` et label de chaque sous-ensemble |
| `INVENTORY_COLUMNS` | 6 colonnes | Colonnes de la table produite par `build_inventory` |
| `REMOVED_COLUMNS` | 6 colonnes | Colonnes du rapport des suppressions |
| `THUMBNAIL_SIZE` | `32` | Côté des miniatures utilisées pour les quasi-doublons, en pixels |
| `NEAR_DUPLICATE_THRESHOLD` | `0.15` | Distance maximale entre miniatures pour un quasi-doublon |

## Inventaire

::: brain_mri.dataset.build_inventory

## Empreintes

::: brain_mri.dataset.file_hash

::: brain_mri.dataset.add_file_hash

## Déduplication

::: brain_mri.dataset.deduplicate_exact

::: brain_mri.dataset.load_thumbnail

::: brain_mri.dataset.find_near_duplicate_pairs

::: brain_mri.dataset.deduplicate_near

::: brain_mri.dataset.DeduplicationResult

::: brain_mri.dataset.deduplicate

## Manifeste

::: brain_mri.dataset.write_manifest

::: brain_mri.dataset.read_manifest

## Ligne de commande

::: brain_mri.dataset.main
