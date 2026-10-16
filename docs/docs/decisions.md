# Décisions

Les choix faits pendant la préparation des données et l'extraction des features, avec leur justification.

## Les données brutes restent immuables

**Décision.** `data/raw/` n'est jamais modifié. Toute transformation produit un fichier dans `data/interim/` ou `data/processed/`.

**Raison.** La préparation reste rejouable et on peut toujours revenir aux images d'origine.

## La logique vit dans `src/`, pas dans les notebooks

**Décision.** Le chargement et la déduplication sont dans `brain_mri.dataset`, testés par `pytest`. Les notebooks importent ces fonctions.

**Raison.** Une même logique n'est écrite qu'une fois, elle est testable, et les notebooks restent des supports d'analyse.

## Règle de conservation des copies

**Décision.** Pour un groupe de copies, on conserve la copie annotée en priorité, sinon l'`image_id` le plus petit.

**Raison.** Ne jamais perdre un label et rendre le choix reproductible. Les copies non annotées d'une image annotée sont retirées pour éviter une fuite entre apprentissage et évaluation.

## Quasi-doublons : miniatures 32 × 32 et seuil de 0,15

**Décision.** Les quasi-doublons sont détectés par distance entre miniatures 32 × 32 en niveaux de gris, avec un seuil de 0,15, et traités avec la même règle que les doublons exacts.

**Raison.** Les paires sous ce seuil sont visuellement identiques, avec une différence absolue quasi nulle en pleine résolution. Le seuil est choisi par inspection et reste une constante documentée.

**Alternative écartée.** Un hash perceptuel (`imagehash`) aurait ajouté une dépendance pour un résultat comparable sur ce jeu.

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

**Raison.** C'est une tache blanche saturée, sans aucun détail anatomique : ses caractéristiques ne seraient que du bruit. Elle est non annotée, donc les 98 images annotées sont conservées. Les autres images atypiques (couleur, valeurs extrêmes) sont de vraies IRM lisibles : les retirer aurait supprimé des images annotées, notamment 23 des 48 `normal` signalées par une règle statistique globale, alors qu'il s'agit d'un décalage de la classe.

## Niveaux de gris et égalisation d'histogramme

**Décision.** Toutes les images sont converties en niveaux de gris, puis égalisées par histogramme en pleine résolution, avant le redimensionnement.

**Raison.** Les couleurs de 10 images sont des artefacts sans information médicale. La luminosité et le contraste suffisent à séparer en grande partie les classes annotées (AUC de 0,82 avec trois descripteurs) : c'est un indice parasite. Après égalisation, cet indice tombe à 0,66 alors que l'AUC des embeddings ne baisse pas (0,978 pour `layer3`, 0,966 pour `layer4`).

**Limites.** Le biais est limité, pas supprimé : il reste la part de fond, et les différences de séquence ou de source d'acquisition ne sont pas corrigées, faute de métadonnées.

## ResNet50 gelé, deux couches, sans rognage

**Décision.** Les features viennent d'un ResNet50 pré-entraîné sur ImageNet, gelé, sur `layer3` et `layer4`, avec un average pooling global. Les images sont redimensionnées à 224 × 224 sans rognage.

**Raison.** La mission recommande de geler les couches convolutives et de comparer plusieurs couches. Les deux couches ne se départagent pas (silhouette, écart cosinus et AUC vont dans des sens différents, avec un écart d'AUC inférieur à l'écart-type) : elles sont conservées. Le redimensionnement sans rognage évite de couper de l'anatomie en bordure.

## Les features sont enregistrées en `.npz` avec les identifiants

**Décision.** Un fichier `.npz` par couche contient les `image_id` et la matrice de features, dans le même ordre.

**Raison.** L'alignement avec le manifeste ne peut pas se perdre, et le format n'ajoute aucune dépendance (NumPy suffit).

## Standardisation et encodage reportés

**Décision.** Les features ne sont pas standardisées à cette étape, et le label n'est encodé (0 / 1) que pour les images annotées, quand un modèle en a besoin.

**Raison.** La standardisation dépend de la population sur laquelle on l'ajuste, donc du jeu de test à définir pour l'étape semi-supervisée : l'ajuster maintenant sur toutes les images mélangerait les données de test. Les images non annotées ne forment pas une classe : elles sont exploitées sans label par la méthode semi-supervisée, et ne reçoivent donc jamais de code.
