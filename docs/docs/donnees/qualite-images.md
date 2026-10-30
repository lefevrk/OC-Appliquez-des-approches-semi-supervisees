# Qualité des images

Notebook : `notebooks/01.02-kl-image-quality.ipynb`

Les notebooks précédents ont vérifié que les fichiers sont lisibles et homogènes. Cette page regarde le **contenu** des images : y a-t-il des particularités qui pourraient fausser la suite ? Les mesures portent sur les 1 506 images brutes ; les descripteurs sont calculés par `brain_mri.quality`.

## Descripteurs

Cinq grandeurs simples sont calculées pour chaque image :

| Descripteur | Définition |
|---|---|
| `brightness` | Luminosité moyenne (0 à 255) |
| `contrast` | Écart-type des niveaux de gris |
| `background_fraction` | Part des pixels quasi noirs (niveau de gris inférieur à 10) |
| `channel_gap` | Écart maximal entre canaux RGB, nul ou presque pour une image en niveaux de gris |
| `sharpness` | Netteté : variance du laplacien des niveaux de gris, en pleine résolution. Plus elle est grande, plus les contours sont marqués |

## Les sous-ensembles se ressemblent-ils ?

| Sous-ensemble | Luminosité médiane | Contraste médian | Part de fond médiane | Netteté médiane |
|---|---:|---:|---:|---:|
| `cancer` | 43,6 | 42,8 | 0,39 | 54,4 |
| `normal` | 55,3 | 56,1 | 0,36 | 115,2 |
| `sans_label` | 50,2 | 44,9 | 0,36 | 54,2 |

Les images `cancer` sont plus sombres, moins contrastées et **deux fois moins nettes** que les images `normal`. La part de fond est voisine. Si ces grandeurs suffisent à deviner le label, un modèle peut les apprendre à la place de la pathologie : c'est un **biais lié à l'image**, traité par le prétraitement (voir [Extraction des features](../pipeline/features.md)).

![Distribution des descripteurs de contenu par sous-ensemble](../assets/figures/descripteurs-images.png)

Les images non annotées se situent entre les deux classes pour la luminosité et le contraste (plus proche des `cancer` pour ce dernier), et ont la netteté des `cancer`. L'hypothèse semi-supervisée suppose que annotées et non annotées viennent de la même distribution : ces écarts invitent à la prudence.

## Images atypiques

Les images aux valeurs extrêmes sont inspectées visuellement.

**En couleur.** 10 images sont en couleur (1 annotée `normal`, 9 non annotées). Les couleurs sont des artefacts : le prétraitement les supprime en convertissant tout en niveaux de gris.

![Images en couleur](../assets/figures/images-atypiques-couleur.png)

**Les plus sombres, les plus lumineuses, avec le plus de fond, les moins nettes.**

![Images les plus sombres](../assets/figures/images-atypiques-sombres.png)

![Images les plus lumineuses](../assets/figures/images-atypiques-lumineuses.png)

![Images avec le plus de fond noir](../assets/figures/images-atypiques-fond.png)

![Images les moins nettes](../assets/figures/images-atypiques-floues.png)

Ce sont de vraies IRM lisibles, simplement traitées ou cadrées différemment : elles sont **conservées**.

## Image exclue

Une seule image est retirée : `ca4c389b-8e9b-40d0-b149-1c81934ea251`, une tache blanche saturée (luminosité de 116,4 et contraste de 119,2, le plus élevé du jeu), avec des artefacts visuels et aucun détail anatomique exploitable. Ses caractéristiques ne seraient que du bruit pour le modèle.

![Image exclue](../assets/figures/image-exclue.png)

- Elle est **non annotée** : les images annotées ne sont pas touchées.
- Le fichier brut n'est pas modifié. L'image est exclue du manifeste par `brain_mri.dataset` (constante `EXCLUDED_IMAGES`) et apparaît dans le rapport des suppressions avec la raison `excluded`. La décision est prise dans le notebook [Doublons](doublons.md), avant la déduplication.

!!! warning "Limites"
    - Ces écarts peuvent venir de différences de source d'acquisition (appareil, protocole) plutôt que de la pathologie. Sans métadonnées, on ne peut pas le vérifier.
    - Les descripteurs sont des mesures globales simples : ils ne remplacent pas une lecture par un radiologue.

!!! note "Valeurs manquantes"
    Un jeu d'images n'a pas de valeurs manquantes au sens d'un tableau. L'équivalent est un fichier absent ou illisible, et les 1 506 images se lisent sans erreur (voir [Exploration](exploration.md)).
