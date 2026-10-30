# Module `brain_mri.modeling.preprocessing`

Standardisation et réduction de dimension des features, ajustées sur un sous-ensemble d'images (l'entraînement) puis appliquées à toutes. Le détail du flux est décrit dans la page [Analyse non supervisée](../pipeline/analyse-non-supervisee.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `RANDOM_STATE` | `42` | Graine des algorithmes aléatoires |
| `N_COMPONENTS` | `50` | Nombre de composantes conservées par la PCA |

## Standardisation

`fit_standardizer` renvoie le `StandardScaler` ajusté, dont on a besoin pour ramener une direction de l'espace réduit vers l'espace des features (voir [Analyse non supervisée](../pipeline/analyse-non-supervisee.md)). `standardize` en applique le résultat à toutes les images.

::: brain_mri.modeling.preprocessing.fit_standardizer

::: brain_mri.modeling.preprocessing.standardize

## Réduction de dimension

`fit_pca` renvoie la PCA ajustée ; `reduce_with_pca` en applique le résultat à toutes les images.

::: brain_mri.modeling.preprocessing.fit_pca

::: brain_mri.modeling.preprocessing.reduce_with_pca
