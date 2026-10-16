# Jeu de données

## Source et description

Le jeu de données est un ensemble d'IRM cérébrales fourni dans le cadre du projet. Il est décrit par le fichier texte placé dans `data/raw/`, qui précise une utilisation libre à des fins académiques.

Les images annotées se répartissent en deux catégories : `normal` (cerveaux sains) et `cancer` (cerveaux présentant des signes de tumeurs).

## Structure

```text
data/raw/
├── sans_label/             # 1 406 images non annotées
└── avec_labels/
    ├── cancer/             #    50 images
    └── normal/             #    50 images
```

| Sous-ensemble | Label | Images |
|---|---|---:|
| `cancer` | `cancer` | 50 |
| `normal` | `normal` | 50 |
| `sans_label` | non annoté | 1 406 |
| **Total** | | **1 506** |

Seules **6,6 %** des images sont annotées. Ce déséquilibre entre données annotées et non annotées est la motivation de l'approche semi-supervisée. Les classes annotées sont en revanche équilibrées (50 / 50).

![Répartition des images : 6,6 % annotées, et classes annotées équilibrées](../assets/figures/repartition-donnees.png)

## Caractéristiques des images

- **Format** : JPEG (`.jpg`)
- **Dimensions** : 512 × 512 pixels pour toutes les images
- **Mode couleur** : RGB

## Aperçu

Cinq images tirées au hasard dans chaque catégorie. Le jeu mélange plusieurs plans de coupe (axial, coronal, sagittal) et plusieurs contrastes, y compris parmi les images annotées.

![Aperçu de cinq IRM non annotées, cancer et normal](../assets/figures/apercu-irm.png)

## Écart avec la description fournie

| Critère | Description fournie | Contenu réel |
|---|---|---:|
| Images non annotées | 1 400 | 1 406 |
| Images annotées | 100 | 100 |
| Total | 1 500 | 1 506 |

Le jeu contient 6 images non annotées de plus que ne l'annonce la description. On retient les chiffres réels pour la suite.

## Annotations

Les labels sont fournis par l'organisation des dossiers : il n'existe pas de fichier de métadonnées séparé. Ils servent à décrire le sous-ensemble connu, puis à l'évaluation et à l'apprentissage supervisé ou semi-supervisé. Les vignettes affichées dans les notebooks sont un outil de contrôle exploratoire : elles ne constituent ni un diagnostic médical ni une validation clinique des labels.
