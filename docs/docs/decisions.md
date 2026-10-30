# Décisions

Les choix faits pendant la préparation des données, l'extraction des features et l'analyse non supervisée, avec leur justification.

## Les données brutes restent immuables

**Décision.** `data/raw/` n'est jamais modifié. Toute transformation produit un fichier dans `data/interim/` ou `data/processed/`.

**Raison.** La préparation reste rejouable et on peut toujours revenir aux images d'origine.

## La logique vit dans `src/`, pas dans les notebooks

**Décision.** Le chargement, la déduplication, l'extraction, la standardisation et les labels faibles sont dans `brain_mri`, testés par `pytest`. Les notebooks importent ces fonctions. Seul le suivi des expériences (`src/utils/mlflow_utils.py`) est appelé uniquement par les notebooks.

**Raison.** Une même logique n'est écrite qu'une fois, elle est testable, et les notebooks restent des supports d'analyse.

## Règle de conservation des copies

**Décision.** Pour un groupe de copies, on conserve la copie annotée en priorité, sinon l'`image_id` le plus petit.

**Raison.** Ne jamais perdre un label et rendre le choix reproductible. Les copies non annotées d'une image annotée sont retirées pour éviter une fuite entre apprentissage et évaluation.

## Quasi-doublons : 8 orientations, miniatures 64 × 64 et seuil de 0,02

**Décision.** Les quasi-doublons sont détectés par l'écart quadratique moyen (RMS) entre miniatures 64 × 64 en niveaux de gris, comparées dans les **8 orientations** (rotations et miroirs), avec un seuil de 0,02, et traités avec la même règle que les doublons exacts.

**Raison.** Comparer des miniatures 32 × 32 dans la seule orientation d'origine, avec un seuil de 0,15, trouve 24 paires et laisse passer toute copie retournée ou pivotée. Sous le seuil de 0,02 et avec les 8 orientations, on trouve 125 paires, dont 94 retournées ou pivotées. Ces copies auraient pu se retrouver une moitié en entraînement, l'autre en test. Les galeries de paires montrent que, sous 0,02, ce sont des copies ; au-delà de 0,03, ce sont surtout des coupes voisines d'un même examen. Le RMS rend le seuil indépendant de la taille de la miniature.

**Limites.** Le seuil est choisi par inspection visuelle, les 41 paires de la zone de 0,02 à 0,04 (dont 5 avec une image annotée) ne sont pas retirées, et les coupes voisines d'un même examen ne sont pas retirées faute de métadonnées de patient.

**Alternative écartée.** Un hash perceptuel (`imagehash`) aurait ajouté une dépendance, sans gérer les retournements.

## Un manifeste CSV avec chemins relatifs

**Décision.** Le jeu dédupliqué est décrit par `data/interim/manifest.csv`, avec des chemins relatifs à `data/raw/`.

**Raison.** Une seule source de vérité pour les pixels, pas de copie de fichiers, un fichier lisible et comparable. Les chemins relatifs le rendent portable.

**Alternatives écartées.**

| Option | Pourquoi pas |
|---|---|
| Copier les images dans `data/interim/` | Duplique les fichiers et ajoute une étape à rejouer |
| Liens symboliques | Fragiles selon le système et les outils |
| Format Parquet | Ajoute `pyarrow` sans gain à cette taille |

## Un rapport des suppressions

**Décision.** Chaque image retirée est listée dans `reports/removed_duplicates.csv`, avec la raison et l'image conservée.

**Raison.** La déduplication reste auditable : on peut vérifier ce qui a été retiré et pourquoi, notamment pour les images annotées.

## Une image inexploitable est exclue

**Décision.** L'image `ca4c389b-8e9b-40d0-b149-1c81934ea251` est exclue du manifeste, via la constante `EXCLUDED_IMAGES` de `brain_mri.dataset`. Le fichier brut reste intact et l'image figure dans le rapport des suppressions avec la raison `excluded`.

**Raison.** C'est une tache blanche saturée, sans aucun détail anatomique : ses caractéristiques ne seraient que du bruit. Elle est non annotée, donc les images annotées ne sont pas touchées. Les autres images atypiques (couleur, valeurs extrêmes, peu nettes) sont de vraies IRM lisibles : les retirer aurait supprimé des images annotées, alors qu'il s'agit d'un décalage de la classe `normal`.

## Niveaux de gris

**Décision.** Toutes les images sont converties en niveaux de gris avant le redimensionnement, sans autre transformation du contenu.

**Raison.** Les couleurs de 10 images du jeu brut (7 dans le manifeste) sont des artefacts sans information médicale (par exemple des bandeaux de texte colorés). La conversion les retire sans rien perdre : le modèle de référence donne la même AUC sur le RGB d'origine et sur les niveaux de gris (0,957), car 1 284 images sur 1 291 sont déjà en niveaux de gris.

**Limites.** Les différences de luminosité, de contraste, de séquence ou de source d'acquisition ne sont pas corrigées, faute de métadonnées. L'analyse non supervisée montre que les clusters suivent en partie le plan de coupe, et que luminosité et contraste restent des indices à surveiller.

## ResNet50 gelé, `layer4` extraite

**Décision.** Les features viennent d'un ResNet50 pré-entraîné sur ImageNet, gelé, sur `layer4` (2 048 valeurs), avec un average pooling global. Les images sont redimensionnées à 224 × 224 sans rognage.

**Raison.** La mission recommande de geler les couches convolutives. `layer4`, la dernière couche convolutive, donne les caractéristiques de plus haut niveau. Le redimensionnement sans rognage évite de couper de l'anatomie en bordure.

## Les features sont enregistrées en `.npz` avec les identifiants

**Décision.** Le fichier `.npz` des features contient les `image_id` et la matrice, dans le même ordre.

**Raison.** L'alignement avec le manifeste ne peut pas se perdre, et le format n'ajoute aucune dépendance (NumPy suffit).

## Standardisation ajustée sur l'entraînement, encodage limité aux annotées

**Décision.** Les features ne sont pas standardisées à l'extraction : la standardisation et la PCA se font à l'étape d'analyse non supervisée, ajustées sur l'**entraînement uniquement**, une fois le découpage défini. Le label n'est encodé (1 pour `cancer`, 0 pour `normal`) que pour les images annotées, quand un calcul en a besoin.

**Raison.** La standardisation dépend de la population sur laquelle on l'ajuste : l'ajuster sur toutes les images mélangerait les données de validation et de test. Les images non annotées ne forment pas une classe : elles sont exploitées sans label par la méthode semi-supervisée, et ne reçoivent donc jamais de code.

## Entraînement, validation et test, définis avant toute transformation

**Décision.** Le test (15 images) et la validation (14 images) sont tirés parmi les images annotées uniquement, de façon stratifiée et avec une graine fixe (`split_dataset`), ce qui laisse 68 annotées à l'entraînement : environ 70 % / 15 % / 15 % des 97 annotées. L'entraînement regroupe ces 68 annotées et toutes les images non annotées. Le découpage est écrit dans `data/processed/split.csv` par `make data`.

**Raison.** Il faut des labels forts pour évaluer, donc test et validation sont annotés. Définir les trois jeux en premier garantit que le test ne participe à aucun ajustement (standardisation, PCA, clustering, choix de paramètres). La validation sert aux vérifications en cours de route et, à l'étape suivante, à l'arrêt anticipé de l'entraînement. Avec 68 annotées à l'entraînement, le vote cluster → classe s'appuie sur plus d'images.

**Limites.** Avec 97 annotées, les jeux restent très petits : une image de test représente 6,7 points d'exactitude et une image de validation 7,1 points. Les mesures sont indicatives.

## K-Means comme référence et seule source de labels faibles

**Décision.** K-Means (couche `layer4`, PCA conservant 90 % de la variance soit 479 composantes, `k = 2`, seed fixe) attribue les labels faibles. DBSCAN est conservé pour la comparaison, sans servir à étiqueter.

**Raison.** K-Means attribue une classe à **toutes** les images, produit directement deux groupes comme les deux classes et ne demande aucun réglage sur 68 images (`k` est fixé à 2). DBSCAN ne forme qu'un seul cluster sur les 12 configurations de la grille : il étiquette `cancer` toutes les images annotées, et son ARI est nul, contre 0,271 pour K-Means. L'ARI de K-Means varie beaucoup d'un jeu à l'autre (0,122 à la validation, 0,005 au test), ce qui reflète la taille de ces jeux (14 et 15 images), et il est faible : les labels faibles sont à utiliser avec prudence.

## La PCA conserve 90 % de la variance

**Décision.** Le nombre de composantes de la PCA n'est pas fixé : on garde le plus petit nombre qui conserve 90 % de la variance de l'entraînement (`VARIANCE_THRESHOLD`), soit 479 composantes pour `layer4`.

**Raison.** Un nombre fixe de composantes n'a pas de justification : à 50 composantes, la PCA ne conserverait que 46 % de la variance de `layer4`. Un seuil de variance rend le choix explicite. L'ARI de K-Means ne dépend pas du nombre de composantes (même valeur pour 10, 50 et 479 composantes).

## Le bruit n'est ni un cluster, ni une classe

**Décision.** Les images que DBSCAN classe en bruit (`-1`) sont rapportées (part des images), mais n'entrent pas dans l'ARI et ne reçoivent aucune classe. Le vote cluster → classe ignore le bruit.

**Raison.** Le bruit est un résultat attendu de cette méthode : il désigne les images qu'aucune zone dense n'accepte. Lui attribuer une classe reviendrait à décider à la main que « dispersé » signifie `normal`, une règle trouvée après coup sur peu d'images.

## La confiance est une marge, pas une probabilité

**Décision.** La confiance d'un label faible est la marge relative entre la distance au centre le plus proche et la distance à l'autre centre, entre 0 et 1.

**Raison.** Elle sert à classer les images selon leur éloignement de la frontière entre clusters. Elle n'est pas calibrée : elle ne dit pas la probabilité que le label soit correct, et son lien avec l'exactitude n'est pas validé. Un éventuel seuil de confiance se choisira à l'étape suivante, par son effet sur le modèle.

## `k = 2` malgré le balayage

**Décision.** On garde deux clusters, alors que l'ARI atteint 0,390 avec `k = 3` (0,271 avec `k = 2`).

**Raison.** Le balayage de `k` de 2 à 6 ne montre aucun coude : l'inertie baisse régulièrement, la silhouette est faible (0,109 pour `k = 2`) et tombe à 0,03 dès `k = 3`, et l'ARI fluctue (de 0,27 à 0,39) sur 68 images annotées. La pureté monte mécaniquement avec `k`. Deux groupes correspondent aux deux classes du jeu annoté.

## Pas de seuil de décision plus prudent

**Décision.** Le label faible suit la règle du centre le plus proche (seuil de 0,50). Un seuil plus élevé, qui étiquette `cancer` davantage d'images, est écarté.

**Raison.** Choisi sur l'entraînement (rappel de `cancer` maximal avec au plus 20 % de fausses alertes), le seuil de 0,54 fait passer le rappel de 0,57 à 0,94 (2 cancers ratés au lieu de 15), au prix de fausses alertes (de 3 % à 18 %). Sur la validation, le rappel passe de 0,57 à 0,86, soit **deux cancers** de plus détectés sur 7, avec une fausse alerte de plus. Un gain de deux images, sur un seuil choisi sur 68 images et vérifié sur 14, ne suffit pas à justifier une règle de plus.

## Cartes d'attention : sans labels, puis directionnelles

**Décision.** À l'extraction, les cartes d'activation et le Grad-CAM de `layer4` du ResNet50 répondent à une seule question, où le réseau regarde, sans labels ni clustering. Après le clustering, des cartes **directionnelles** montrent ce qui distingue les deux groupes : la direction du centre `normal` au centre `cancer` est ramenée dans l'espace des features (PCA inverse, puis division par l'écart-type de la standardisation), et la carte est centrée sur le milieu des deux centres.

**Raison.** Le pooling global étant une moyenne, la moyenne de la carte est exactement le score de l'image sur la direction, ce que le notebook vérifie. Sans centrage sur la frontière, le rouge et le bleu seraient arbitraires. Le Grad-CAM porte sur une classe d'ImageNet, sans sens médical : il se lit comme « les zones qui comptent pour le réseau », pas comme un indice de cancer.

**Limites.** Les cartes sont grossières (7 × 7 pour `layer4`) et ne localisent pas la lésion ; elles semblent suivre le plan de coupe autant que la pathologie.

## Suivi des expériences MLflow en local

**Décision.** Les clusterings sont suivis avec MLflow, en local (base SQLite `mlflow.db` et dossier `mlruns/`, ignorés par git), dans une seule expérience, avec un run à plat par configuration testée. Seuls les notebooks qui comparent des configurations créent des runs.

**Raison.** Chaque valeur de `k`, chaque configuration de DBSCAN et le K-Means retenu sont comparables dans l'interface, avec leurs paramètres, leurs métriques, leur jeu d'entraînement, leurs figures et leurs modèles ajustés. Les tags (`git_commit`, `manifest_sha256`, `seed`...) relient chaque run au code et aux données qui l'ont produit. Voir [Conventions](conventions.md#suivi-des-experiences).

## Aucun chemin personnel dans les sorties de notebooks

**Décision.** Les sorties de notebook n'affichent jamais le chemin de la machine : `mask_path` le remplace par `<PROJECT_ROOT>` ou `~`, et un test échoue si un notebook en contient un.

**Raison.** Les notebooks sont versionnés et publiés : un chemin personnel n'a rien à y faire, et le test évite qu'il y revienne.
