# Analyse non supervisée

Notebook : `notebooks/03.01-kl-unsupervised-analysis.ipynb`

Les features de l'étape précédente servent à **regrouper les images**, puis à attribuer un **label faible** aux images non annotées. Le découpage est implémenté dans `brain_mri.dataset`, la standardisation et la PCA dans `brain_mri.modeling.preprocessing`, le clustering et les labels faibles dans `brain_mri.modeling.unsupervised`. Les labels faibles se produisent avec `make unsupervised`.

Chaque clustering testé est suivi dans MLflow (un run par configuration, voir [Conventions](../conventions.md#suivi-des-experiences)) : `make mlflow-ui`.

## Flux

```mermaid
flowchart TB
    MAN[/data/interim/manifest.csv/] --> SPLIT[split_dataset]
    SPLIT --> SPL[/data/processed/split.csv/]
    FEAT[/features_resnet50_layer4.npz/] --> STD[fit_standardizer]
    SPL --> STD
    STD --> PCA[fit_pca : 90 % de la variance]
    PCA --> KM[K-Means, k = 2]
    PCA --> DENS[DBSCAN : comparaison]
    KM --> MAP[Vote cluster → classe]
    MAP --> WEAK[/data/processed/weak_labels.csv/]
```

## Trois jeux, définis avant toute transformation

La standardisation, la PCA et le clustering **s'ajustent sur des données**. Si le jeu de test participait à cet ajustement, il ne serait plus « jamais vu ». Le [découpage](preparation.md) est donc fait en premier :

| Jeu | Rôle |
|---|---|
| **Entraînement** (1 262 images : 68 annotées et 1 194 non annotées) | Ajuste la standardisation, la PCA et les clusterings, et sert au vote cluster → classe |
| **Validation** (14 images annotées) | Vérifications en cours de route ; à l'étape suivante, arrêt anticipé de l'entraînement |
| **Test** (15 images annotées) | Évaluation finale |

## Standardisation et PCA

Chaque dimension est centrée et réduite avec le `StandardScaler` ajusté sur l'**entraînement uniquement**, puis la PCA, ajustée sur le même jeu, garde **le plus petit nombre de composantes qui conserve 90 % de la variance** (`VARIANCE_THRESHOLD`) : **479** composantes pour `layer4`. Le t-SNE ne sert qu'à la visualisation, sur l'entraînement.

Hors entraînement (images annotées uniquement), la moyenne des features standardisées s'écarte de 0 (0,171 en valeur absolue moyenne par dimension) et les écarts-types sont plus petits (0,854) : les images annotées ne suivent pas exactement la distribution des non annotées.

![Variance expliquée cumulée par la PCA](../assets/figures/variance-expliquee-pca.png)

![Projections PCA et t-SNE](../assets/figures/reduction-2d-labels.png)

Sur la PCA, les `normal` annotées forment un groupe compact à gauche, alors que les `cancer` se répartissent de la gauche vers la droite, avec un recouvrement dans la zone des `normal`. Sur le t-SNE, les `normal` se regroupent en haut à gauche et les `cancer` sont plus dispersées.

## Clustering

On cherche **deux groupes**, comme dans le jeu annoté. La qualité se mesure par l'**ARI** (*Adjusted Rand Index*) entre les clusters et les labels forts : 1 pour un accord parfait, 0 pour un accord équivalent au hasard. Il est calculé sur les 68 images annotées de l'entraînement, avec très peu d'images : les valeurs sont **indicatives**.

| Méthode | Principe | Paramètres |
|---|---|---|
| **K-Means** | Deux groupes compacts autour de deux centres | `k = 2`, 10 initialisations, seed fixe |
| **DBSCAN** | Zones denses, le reste est du bruit | `min_samples`, `eps` (quantiles de la courbe des k plus proches voisins) |

Le **bruit** (cluster `-1`) est un résultat attendu de DBSCAN. Il est rapporté, mais il n'est **ni un cluster, ni une classe** : l'ARI est calculé sur les images annotées qui appartiennent à un cluster.

### Résultats sur `layer4`

Pour DBSCAN, la configuration retenue est celle de la grille (12 configurations) qui a au moins deux clusters, au plus 30 % de bruit et le meilleur ARI hors bruit. Si aucune configuration ne forme deux clusters avec peu de bruit, on garde celle qui forme le plus de clusters, puis celle qui a le moins de bruit. Elle sert à illustrer la méthode.

| Méthode | Clusters | Couverture à l'entraînement | ARI entraînement | ARI validation | ARI test |
|---|---:|---:|---:|---:|---:|
| K-Means | 2 | 100 % | 0,271 | 0,122 | 0,005 |
| DBSCAN | 1 | 97 % | 0,000 | non évalué | non évalué |

- **K-Means est stable** : même ARI pour 10 et 50 composantes et pour 479 composantes (0,271), 0,240 pour les features complètes, et un ARI minimal de 0,925 entre 10 seeds. Un ARI d'environ 0,3 est meilleur que le hasard (0) mais loin d'un accord parfait (1), et l'ARI de test (0,005) ne se distingue pas du hasard sur 15 images.
- **DBSCAN** suppose des zones denses, ce que le nuage de points n'offre pas : les 12 configurations de la grille ne forment qu'un seul cluster, avec de 3 % à 24 % d'images en bruit. La configuration retenue étiquette `cancer` toutes les images annotées, et son ARI est nul.

![Clusters de chaque méthode sur la PCA et le t-SNE](../assets/figures/clusters-2d.png)

Sur les annotées de l'entraînement, K-Means rate **15 `cancer` sur 35** et prend **1 `normal` sur 33** pour un `cancer`. DBSCAN, qui ne forme qu'un cluster, étiquette `cancer` les 35 `cancer` mais aussi les 33 `normal`.

![Matrices de confusion de K-Means et de DBSCAN](../assets/figures/matrices-confusion-methodes.png)

### Nombre de clusters

Le jeu annoté compte deux classes, mais rien n'oblige les images à former deux groupes. On fait varier `k` de 2 à 6 sur `layer4` :

| `k` | Inertie | Silhouette | ARI | Pureté |
|---:|---:|---:|---:|---:|
| **2** | 2 195 839 | 0,109 | 0,271 | 0,765 |
| 3 | 2 132 070 | 0,034 | 0,390 | 0,853 |
| 4 | 2 087 941 | 0,025 | 0,296 | 0,838 |
| 5 | 2 050 639 | 0,043 | 0,281 | 0,838 |
| 6 | 2 023 002 | 0,003 | 0,361 | 0,897 |

Aucun nombre de clusters ne se détache : l'inertie baisse régulièrement sans coude net, la silhouette est faible et tombe à 0,03 dès `k = 3`, et l'ARI fluctue. La pureté monte avec `k` par construction, car des clusters plus petits sont plus faciles à rendre homogènes. Avec `k = 3`, la PCA est découpée en trois bandes successives sur la composante 1 ; l'ARI (0,39 contre 0,27) progresse, mais la silhouette s'effondre et l'ARI repasse à 0,30 pour `k = 4`. **On garde `k = 2`**, qui correspond aux deux classes du jeu annoté.

![K-Means selon le nombre de clusters](../assets/figures/balayage-k.png)

![K-Means avec k = 2 et k = 3 sur la PCA et le t-SNE](../assets/figures/k2-contre-meilleur-k.png)

### Seuil de décision

En contexte médical, rater un cancer est plus grave qu'une fausse alerte. On teste une règle plus prudente que « centre le plus proche » : étiqueter `cancer` quand le rapport `d_cancer / (d_cancer + d_normal)` est inférieur à un seuil supérieur à 0,5. Le seuil est choisi sur les images annotées de l'**entraînement** (rappel de `cancer` maximal avec au plus 20 % de fausses alertes), la validation le vérifie et le test n'est pas utilisé.

| Jeu | Seuil | Rappel `cancer` | Fausses alertes | ARI |
|---|---:|---:|---:|---:|
| Entraînement | 0,50 | 0,571 | 3,0 % | 0,271 |
| Entraînement | 0,54 | 0,943 | 18,2 % | 0,579 |
| Validation | 0,50 | 0,571 | 14,3 % | 0,122 |
| Validation | 0,54 | 0,857 | 28,6 % | 0,272 |

Le gain se confirme sur la validation, mais il tient à **deux cancers de plus détectés sur 7**, avec une fausse alerte de plus (2 `normal` sur 7 au lieu d'une). Un seuil choisi sur 68 images et vérifié sur 14 est trop fragile pour changer la règle : on garde la règle de K-Means, au seuil de 0,50.

![Effet du seuil de décision sur le label faible](../assets/figures/seuil-decision.png)

## Labels faibles

Un cluster n'a pas de sens médical : on attribue à chaque cluster de K-Means la classe **majoritaire** parmi ses images annotées de l'entraînement (égalité : `normal`), puis ce label faible à toutes les images du cluster. Seul K-Means fournit des labels faibles : les images laissées en bruit par DBSCAN n'auraient pas de classe.

![K-Means, centres compris, et labels forts sur la PCA de layer4](../assets/figures/kmeans-clusters-labels.png)

### Confiance

Chaque label faible porte une **confiance** `(d2 - d1) / (d2 + d1)`, où `d1` est la distance au centre le plus proche et `d2` la distance à l'autre centre. Elle vaut 0 à égale distance des deux centres et 1 sur le centre. C'est une **marge géométrique, pas une probabilité** d'avoir le bon label, et elle n'est pas validée.

### Fichier produit : `data/processed/weak_labels.csv`

Une ligne par image **non annotée**.

| Colonne | Contenu |
|---|---|
| `image_id` | L'image |
| `cluster` | Cluster de K-Means |
| `weak_label` | `cancer` ou `normal` |
| `confidence` | Marge entre les deux centres, entre 0 et 1 |
| `method` | `K-Means` |

Les 1 194 images reçoivent **512 `cancer` et 682 `normal`** (43 % de `cancer`, contre 51,5 % parmi les annotées de l'entraînement). La répartition réelle des non annotées est inconnue : ce rapprochement ne prouve rien, il signale seulement un écart.

La confiance est faible : médiane de 0,04 pour `cancer` et de 0,08 pour `normal`, maximum de 0,32. Sur les 1 194 labels faibles, 878 sont indécis (confiance sous 0,10), 203 hésitants et **113 seulement confiants** (0,20 et plus), tous `normal` : les 512 labels faibles `cancer` sont tous proches de la frontière entre les deux clusters.

![Confiance des labels faibles](../assets/figures/confiance-labels-faibles.png)

Les images les plus confiantes du label `cancer` sont toutes des coupes sagittales, celles du label `normal` des coupes axiales. Deux des huit images du label `normal` paraissent pourtant montrer une masse visible : la confiance mesure la **proximité du centre**, pas l'absence de tumeur. Le regroupement semble suivre le plan de coupe autant que la pathologie.

![Images les plus confiantes de chaque label faible](../assets/figures/images-confiantes.png)

### Qualité sur des images jamais vues

| Jeu | Images | Exactitude | Intervalle de confiance à 95 % | Exactitude équilibrée |
|---|---:|---:|---|---:|
| Entraînement (optimiste : le vote utilise ces images) | 68 | 0,765 | | |
| Validation | 14 | 0,714 | [0,454 ; 0,883] | 0,714 |
| Test | 15 | 0,600 | [0,357 ; 0,802] | 0,625 |

| Jeu | Classe | Précision | Rappel | F1 | Images |
|---|---|---:|---:|---:|---:|
| Validation | `cancer` | 0,800 | 0,571 | 0,667 | 7 |
| Validation | `normal` | 0,667 | 0,857 | 0,750 | 7 |
| Test | `cancer` | 1,000 | 0,250 | 0,400 | 8 |
| Test | `normal` | 0,538 | 1,000 | 0,700 | 7 |

Sur les 29 images jamais vues, le label faible se trompe **10 fois** : 9 `cancer` étiquetées `normal` (3 sur 7 à la validation, 6 sur 8 au test) et une `normal` étiquetée `cancer` (à la validation). Sur les 68 annotées de l'entraînement, il se trompe 16 fois (15 `cancer` étiquetées `normal` et une `normal` étiquetée `cancer`), une mesure optimiste. Ces chiffres ne valent que pour les images annotées : on ne sait pas comment K-Means se comporte sur les 1 194 autres.

![Matrices de confusion du label faible sur la validation et le test](../assets/figures/matrices-confusion-labels-faibles.png)

Sur la PCA, les erreurs de la validation et du test se trouvent surtout dans la partie gauche, du côté où les `normal` annotées sont nombreuses : ce sont des `cancer` dont l'image ressemble à celles des `normal`.

![Erreurs du label faible sur la PCA et le t-SNE, entraînement et images jamais vues](../assets/figures/erreurs-labels-faibles.png)

![Images mal étiquetées par K-Means](../assets/figures/images-mal-etiquetees.png)

### Que capturent les clusters ?

Des indices d'acquisition (luminosité, contraste, part de fond noir) séparent en partie `cancer` et `normal` (voir [Qualité des images](../donnees/qualite-images.md)). Les clusters pourraient suivre ces indices plutôt que la pathologie. On mesure l'AUC de chaque descripteur des **images d'origine** pour prédire le label (0,5 : aucune séparation), sur les **mêmes 68 images annotées** pour les labels forts et les labels faibles : la comparaison n'a de sens que sur les mêmes images. Les intervalles à 95 % viennent d'un rééchantillonnage.

| Descripteur | Label fort | Label faible K-Means (mêmes 68) | Label faible K-Means (1 262 images) |
|---|---|---|---:|
| Luminosité | 0,334 [0,20 ; 0,46] | 0,448 [0,31 ; 0,58] | 0,490 |
| Contraste | 0,201 [0,09 ; 0,31] | 0,309 [0,19 ; 0,44] | 0,422 |
| Part de fond | 0,552 [0,41 ; 0,68] | 0,414 [0,28 ; 0,55] | 0,438 |

Les labels faibles suivent la luminosité et le contraste un peu moins que les labels forts, avec des intervalles qui se recouvrent largement : **on ne peut pas montrer que le clustering exploite moins ces indices d'acquisition que la pathologie**. Pour la part de fond, les AUC diffèrent mais les intervalles se recouvrent aussi. L'AUC sur 1 262 images, plus proche de 0,5, ne se compare pas aux labels forts, mesurés sur 68 images.

### Que regardent les deux groupes ?

Les deux clusters ont chacun un centre dans l'espace de la PCA. La **direction** qui va du centre `normal` au centre `cancer` indique ce qui distingue les deux groupes. On la ramène dans l'espace des caractéristiques (PCA inverse, puis division par l'écart-type de la standardisation). Le pooling global étant une moyenne des cartes spatiales de `layer4` (7 × 7), le score d'une image sur cette direction est la **moyenne** d'une carte obtenue en sommant les 2 048 cartes pondérées par la direction : le calcul est vérifié sur une image (score direct et moyenne de la carte identiques).

La carte est **centrée sur le milieu des deux centres**, c'est-à-dire sur la frontière entre les clusters : un score positif est du côté `cancer` (rouge), un score négatif du côté `normal` (bleu). Sans ce centrage, le rouge et le bleu seraient arbitraires. La limite de couleur est commune à toutes les images pour une direction donnée. On compare avec la direction des **labels forts** de l'entraînement (similarité cosinus de 0,80).

![Cartes directionnelles : annotées bien et mal étiquetées, labels faibles très confiants et incertains](../assets/figures/cartes-directionnelles.png)

- **Zones.** Les zones rouges se trouvent surtout dans la partie basse de la tête (base du crâne, visage) sur les coupes sagittales et coronales ; sur les coupes axiales, les cartes sont presque neutres, avec de faibles zones bleues. **Les cartes ne délimitent pas la lésion** : elles sont grossières (7 × 7, agrandie) et semblent suivre la position dans l'image plus que la structure visible.
- **Plan de coupe.** Les scores les plus élevés côté `cancer` sont des coupes sagittales, les plus bas côté `normal` des coupes axiales. Les deux images mal étiquetées vont dans ce sens : la `cancer` classée `normal` est une coupe axiale, la `normal` classée `cancer` une coupe coronale. Sans métadonnées, on ne peut pas le vérifier.
- **Deux directions.** Elles donnent des cartes proches pour les images bien étiquetées et divergent pour la `cancer` mal étiquetée (score de K-Means de −286, score des labels forts de −17) et pour les images au label faible incertain, dont le score de K-Means est proche de 0 : leur carte est alors pâle.

!!! warning "Limites"
    - Le choix des méthodes repose sur 68 images annotées et leur évaluation sur 14 et 15 : les écarts de quelques points ne sont pas significatifs.
    - Les images annotées ne suivent pas exactement la distribution des non annotées : ce que les 68 annotées disent des 1 194 autres est limité.
    - Les labels faibles ont une confiance faible, une proportion de `cancer` non vérifiable, et les clusters suivent en partie le plan de coupe.
    - Les cartes directionnelles sont grossières (7 × 7) et ne localisent pas la lésion.

## Lancer le calcul

```bash
make unsupervised
```

La commande lit le manifeste, le découpage et les features de `layer4`, ajuste la standardisation, la PCA et K-Means sur l'entraînement, puis écrit `data/processed/weak_labels.csv`. Les valeurs de `confidence` peuvent différer d'environ 10⁻⁷ d'une exécution à l'autre (calcul en `float32`) ; les clusters et les labels faibles sont identiques.

```bash
uv run python -m brain_mri.modeling.unsupervised \
    --manifest-path data/interim/manifest.csv \
    --split-path data/processed/split.csv \
    --features-dir data/processed \
    --output-path data/processed/weak_labels.csv \
    --layer layer4
```

La grille DBSCAN, le balayage de `k`, le seuil de décision, les erreurs et les cartes directionnelles sont dans le notebook.
