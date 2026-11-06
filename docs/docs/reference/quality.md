# Module `brain_mri.quality`

Descripteurs simples du **contenu** des images (luminosité, contraste, netteté, part de fond, couleur), pour repérer des particularités qui pourraient fausser la suite. Le détail est décrit dans la page [Qualité des images](../donnees/qualite-images.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `BACKGROUND_LEVEL` | `10` | Niveau de gris en dessous duquel un pixel est compté comme du fond noir |
| `COLOR_GAP_THRESHOLD` | `2` | Écart maximal entre canaux RGB au-delà duquel une image est dite en couleur |
| `DESCRIPTOR_COLUMNS` | 5 colonnes | `brightness`, `contrast`, `background_fraction`, `channel_gap`, `sharpness` |

## Descripteurs

::: brain_mri.quality.describe_image

::: brain_mri.quality.describe_images
