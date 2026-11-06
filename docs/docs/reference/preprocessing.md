# Module `brain_mri.modeling.preprocessing`

Standardisation et réduction de dimension des features, ajustées sur un sous-ensemble d'images (l'entraînement) puis appliquées à toutes. Le détail du flux est décrit dans la page [Analyse non supervisée](../pipeline/analyse-non-supervisee.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `RANDOM_STATE` | `42` | Graine des algorithmes aléatoires |
| `VARIANCE_THRESHOLD` | `0.90` | Part de la variance que la PCA doit conserver ; le nombre de composantes en découle |

## Standardisation

`fit_standardizer` renvoie le `StandardScaler` ajusté, dont on a besoin pour ramener une direction de l'espace réduit vers l'espace des features (voir [Analyse non supervisée](../pipeline/analyse-non-supervisee.md)). `standardize` en applique le résultat à toutes les images.

::: brain_mri.modeling.preprocessing.fit_standardizer

::: brain_mri.modeling.preprocessing.standardize

## Réduction de dimension

`fit_pca` renvoie la PCA ajustée ; `reduce_with_pca` en applique le résultat à toutes les images. Le paramètre `n_components` est un nombre de composantes (entier) ou une part de variance à conserver (réel entre 0 et 1, `VARIANCE_THRESHOLD` par défaut).

::: brain_mri.modeling.preprocessing.fit_pca

::: brain_mri.modeling.preprocessing.reduce_with_pca
