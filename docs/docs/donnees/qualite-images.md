# Qualité des images

Notebook : `notebooks/03.0-kl-feature-extraction.ipynb` (section 2)

Les notebooks précédents ont vérifié que les fichiers sont lisibles et homogènes. Cette page regarde le **contenu** des images : y a-t-il des particularités qui pourraient fausser la suite ?

## Descripteurs

Quatre grandeurs simples sont calculées pour chaque image :

| Descripteur | Définition |
|---|---|
| `brightness` | Luminosité moyenne (0 à 255) |
| `contrast` | Écart-type des niveaux de gris |
| `background_fraction` | Part des pixels quasi noirs (niveau de gris inférieur à 10) |
| `channel_gap` | Écart maximal entre canaux RGB, nul ou presque pour une image en niveaux de gris |

## Les sous-ensembles se ressemblent-ils ?

| Sous-ensemble | Luminosité médiane | Contraste médian | Part de fond médiane |
|---|---:|---:|---:|
| `cancer` | 43,6 | 42,8 | 0,39 |
| `normal` | 55,3 | 56,1 | 0,36 |
| `sans_label` | 49,8 | 44,4 | 0,36 |

Les images `cancer` sont plus sombres et moins contrastées que les images `normal`. Si ces deux grandeurs suffisent à deviner le label, un modèle peut les apprendre à la place de la pathologie : c'est un **biais lié à l'image**, traité par le prétraitement (voir [Extraction des features](../pipeline/features.md)).

![Distribution de la luminosité, du contraste et de la part de fond par sous-ensemble](../assets/figures/descripteurs-images.png)

En luminosité et en contraste, les images non annotées se situent entre les deux classes annotées.

## Images atypiques

Les images aux valeurs extrêmes sont inspectées visuellement.

**En couleur.** 10 images sont en couleur (1 annotée `normal`, 9 non annotées). Les couleurs sont des artefacts : le prétraitement les supprime en convertissant tout en niveaux de gris.

![Images en couleur](../assets/figures/images-atypiques-couleur.png)

**Les plus sombres, les plus lumineuses, avec le plus de fond.**

![Images les plus sombres](../assets/figures/images-atypiques-sombres.png)

![Images les plus lumineuses](../assets/figures/images-atypiques-lumineuses.png)

![Images avec le plus de fond noir](../assets/figures/images-atypiques-fond.png)

Ce sont de vraies IRM lisibles, simplement traitées ou cadrées différemment : elles sont **conservées**.

## Image exclue

Une seule image est retirée : `ca4c389b-8e9b-40d0-b149-1c81934ea251`, une tache blanche saturée, avec des artefacts visuels et aucun détail anatomique exploitable. Ses caractéristiques ne seraient que du bruit pour le modèle.

- Elle est **non annotée** : les 98 images annotées sont toutes conservées.
- Le fichier brut n'est pas modifié. L'image est exclue du manifeste par `brain_mri.dataset` (constante `EXCLUDED_IMAGES`) et apparaît dans le rapport des suppressions avec la raison `excluded`.

!!! note "Valeurs manquantes"
    Un jeu d'images n'a pas de valeurs manquantes au sens d'un tableau. L'équivalent est un fichier absent ou illisible, et les 1 506 images se lisent sans erreur (voir [Exploration](exploration.md)).
