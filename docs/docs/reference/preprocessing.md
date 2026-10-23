# Module `brain_mri.modeling.preprocessing`

Standardisation et réduction de dimension des features, ajustées sur un sous-ensemble d'images (l'entraînement) puis appliquées à toutes. Le détail du flux est décrit dans la page [Analyse non supervisée](../pipeline/analyse-non-supervisee.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `RANDOM_STATE` | `42` | Graine des algorithmes aléatoires |
| `N_COMPONENTS` | `50` | Nombre de composantes conservées par la PCA |

## Standardisation

::: brain_mri.modeling.preprocessing.standardize

## Réduction de dimension

::: brain_mri.modeling.preprocessing.reduce_with_pca
