# Décisions

Les choix faits pendant la préparation des données, avec leur justification.

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
