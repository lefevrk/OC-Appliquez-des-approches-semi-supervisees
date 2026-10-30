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

**Raison.** La première version comparait des miniatures 32 × 32 dans la seule orientation d'origine, avec un seuil de 0,15 : elle trouvait 24 paires et laissait passer toute copie retournée ou pivotée. Sous le seuil de 0,02, on trouve 125 paires, dont 94 retournées ou pivotées. Elles passaient entre les mailles et auraient pu se retrouver une moitié en entraînement, l'autre en test. Les galeries de paires montrent que, sous 0,02, ce sont des copies ; au-delà de 0,03, ce sont surtout des coupes voisines d'un même examen. Le RMS rend le seuil indépendant de la taille de la miniature.

**Limites.** Le seuil est choisi par inspection visuelle, la zone de 0,02 à 0,04 (41 paires, dont 5 avec une image annotée) reste ouverte, et les coupes voisines d'un même examen ne sont pas retirées faute de métadonnées de patient.

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

## Niveaux de gris et égalisation d'histogramme

**Décision.** Toutes les images sont converties en niveaux de gris, puis égalisées par histogramme en pleine résolution, avant le redimensionnement.

**Raison.** Les couleurs de 10 images du jeu brut (7 dans le manifeste) sont des artefacts sans information médicale. La luminosité et le contraste suffisent à séparer en grande partie les classes annotées (AUC de 0,82 avec trois descripteurs) : c'est un indice parasite. Après égalisation, cet indice tombe à 0,65 alors que l'AUC des features ne baisse pas (0,983 pour `layer3`, 0,977 pour `layer4`).

**Limites.** Le biais est limité, pas supprimé : il reste la part de fond, et les différences de séquence ou de source d'acquisition ne sont pas corrigées, faute de métadonnées. L'analyse non supervisée le confirme : les clusters suivent la luminosité et le contraste autant que les labels forts.

## ResNet18 gelé, deux couches extraites, `layer4` retenue

**Décision.** Les features viennent d'un ResNet18 pré-entraîné sur ImageNet, gelé, sur `layer3` (256 valeurs) et `layer4` (512 valeurs), avec un average pooling global. Les images sont redimensionnées à 224 × 224 sans rognage. `layer4` est retenue pour la suite.

**Raison.** La mission recommande de geler les couches convolutives et de comparer plusieurs couches. Les deux ne se départagent pas sur les images annotées (silhouette, écart cosinus et AUC vont dans des sens différents, avec un écart d'AUC inférieur à l'écart-type) ; `layer4` est choisie, et K-Means y obtient un meilleur accord avec les labels forts (ARI de 0,337 contre 0,092). Le redimensionnement sans rognage évite de couper de l'anatomie en bordure.

## Les features sont enregistrées en `.npz` avec les identifiants

**Décision.** Un fichier `.npz` par couche contient les `image_id` et la matrice de features, dans le même ordre.

**Raison.** L'alignement avec le manifeste ne peut pas se perdre, et le format n'ajoute aucune dépendance (NumPy suffit).

## Standardisation ajustée sur l'entraînement, encodage limité aux annotées

**Décision.** Les features ne sont pas standardisées à l'extraction : la standardisation et la PCA se font à l'étape d'analyse non supervisée, ajustées sur l'**entraînement uniquement**, une fois le découpage défini. Le label n'est encodé (1 pour `cancer`, 0 pour `normal`) que pour les images annotées, quand un calcul en a besoin.

**Raison.** La standardisation dépend de la population sur laquelle on l'ajuste : l'ajuster sur toutes les images mélangerait les données de validation et de test. Les images non annotées ne forment pas une classe : elles sont exploitées sans label par la méthode semi-supervisée, et ne reçoivent donc jamais de code.

## Entraînement, validation et test, définis avant toute transformation

**Décision.** Le test (15 images) et la validation (14 images) sont tirés parmi les images annotées uniquement, de façon stratifiée et avec une graine fixe (`split_dataset`), ce qui laisse 68 annotées à l'entraînement : environ 70 % / 15 % / 15 % des 97 annotées. L'entraînement regroupe ces 68 annotées et toutes les images non annotées. Le découpage est écrit dans `data/processed/split.csv` par `make data`.

**Raison.** Il faut des labels forts pour évaluer, donc test et validation sont annotés. Définir les trois jeux en premier garantit que le test ne participe à aucun ajustement (standardisation, PCA, clustering, choix de paramètres). La validation sert aux vérifications en cours de route et, à l'étape suivante, à l'arrêt anticipé de l'entraînement. Avec 68 annotées à l'entraînement, le vote cluster → classe s'appuie sur plus d'images.

**Limites.** Avec 97 annotées, les jeux restent très petits : une image de test représente 6,7 points d'exactitude et une image de validation 7,1 points. Les mesures sont indicatives.

## K-Means comme référence et seule source de labels faibles

**Décision.** K-Means (couche `layer4`, 50 composantes de PCA, `k = 2`, seed fixe) attribue les labels faibles. DBSCAN est conservé pour la comparaison, sans servir à étiqueter.

**Raison.** K-Means attribue une classe à **toutes** les images, produit directement deux groupes comme les deux classes et n'a aucun paramètre à choisir sur 68 images. La configuration retenue de DBSCAN laisse 16 % des images sans cluster, donc sans classe, et son ARI (0,266, sur 47 des 68 images annotées) reste inférieur à celui de K-Means (0,337). L'ARI de K-Means varie beaucoup d'un jeu à l'autre (0,122 à la validation, 0,504 au test), ce qui reflète la taille de ces jeux (14 et 15 images).

## Le bruit n'est ni un cluster, ni une classe

**Décision.** Les images que DBSCAN classe en bruit (`-1`) sont rapportées (part des images), mais n'entrent pas dans l'ARI et ne reçoivent aucune classe. Le vote cluster → classe ignore le bruit.

**Raison.** Le bruit est un résultat attendu de cette méthode : il désigne les images qu'aucune zone dense n'accepte. Lui attribuer une classe reviendrait à décider à la main que « dispersé » signifie `normal`, une règle trouvée après coup sur peu d'images.

## La confiance est une marge, pas une probabilité

**Décision.** La confiance d'un label faible est la marge relative entre la distance au centre le plus proche et la distance à l'autre centre, entre 0 et 1.

**Raison.** Elle sert à classer les images selon leur éloignement de la frontière entre clusters. Elle n'est pas calibrée : elle ne dit pas la probabilité que le label soit correct, et son lien avec l'exactitude n'est pas validé. Un éventuel seuil de confiance se choisira à l'étape suivante, par son effet sur le modèle.

## `k = 2` malgré le balayage

**Décision.** On garde deux clusters, alors que l'ARI atteint 0,357 avec `k = 3` (0,337 avec `k = 2`).

**Raison.** Le balayage de `k` de 2 à 6 ne montre aucun coude : l'inertie baisse régulièrement, la silhouette reste voisine de 0,12 et l'ARI fluctue (de 0,21 à 0,36) sur 68 images annotées. La pureté monte mécaniquement avec `k`. Deux groupes correspondent aux deux classes du jeu annoté.

## Pas de seuil de décision plus prudent

**Décision.** Le label faible suit la règle du centre le plus proche (seuil de 0,50). Un seuil plus élevé, qui étiquette `cancer` davantage d'images, est écarté.

**Raison.** Choisi sur l'entraînement (rappel de `cancer` maximal avec au plus 20 % de fausses alertes), le seuil de 0,53 fait passer le rappel de 0,63 à 0,71, mais l'exactitude reste à 0,79 et les fausses alertes passent de 3 % à 12 %. Sur la validation, rien ne change (rappel de 0,57, fausses alertes de 14 %). Le gain ne se confirme pas sur des images jamais vues.

## Cartes d'attention : sans labels, puis directionnelles

**Décision.** À l'extraction, les cartes d'activation et le Grad-CAM de `layer4` répondent à une seule question, où le réseau regarde, sans labels ni clustering. Après le clustering, des cartes **directionnelles** montrent ce qui distingue les deux groupes : la direction du centre `normal` au centre `cancer` est ramenée dans l'espace des features (PCA inverse, puis division par l'écart-type de la standardisation), et la carte est centrée sur le milieu des deux centres.

**Raison.** Le pooling global étant une moyenne, la moyenne de la carte est exactement le score de l'image sur la direction, ce que le notebook vérifie. Sans centrage sur la frontière, le rouge et le bleu seraient arbitraires. Le Grad-CAM porte sur une classe d'ImageNet, sans sens médical : il se lit comme « les zones qui comptent pour le réseau », pas comme un indice de cancer.

**Limites.** Les cartes sont grossières (7 × 7 pour `layer4`) et ne localisent pas la lésion ; elles semblent suivre le plan de coupe autant que la pathologie.

## Suivi des expériences MLflow en local

**Décision.** Les clusterings sont suivis avec MLflow, en local (base SQLite `mlflow.db` et dossier `mlruns/`, ignorés par git), dans une seule expérience, avec un run à plat par configuration testée. Seuls les notebooks qui comparent des configurations créent des runs.

**Raison.** Chaque valeur de `k`, chaque configuration de DBSCAN et le K-Means retenu sont comparables dans l'interface, avec leurs paramètres, leurs métriques, leur jeu d'entraînement, leurs figures et leurs modèles ajustés. Les tags (`git_commit`, `manifest_sha256`, `seed`...) relient chaque run au code et aux données qui l'ont produit. Voir [Conventions](conventions.md#suivi-des-experiences).

## Aucun chemin personnel dans les sorties de notebooks

**Décision.** Les sorties de notebook n'affichent jamais le chemin de la machine : `mask_path` le remplace par `<PROJECT_ROOT>` ou `~`, et un test échoue si un notebook en contient un.

**Raison.** Les notebooks sont versionnés et publiés : un chemin personnel n'a rien à y faire, et le test évite qu'il y revienne.
