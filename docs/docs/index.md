# BrainScanAI - IRM cérébrales semi-supervisées

Exploration et détection semi-supervisée de tumeurs cérébrales à partir d'IRM.

Le projet exploite des images IRM **partiellement annotées** : seules 100 images sur 1 506 portent un label (`cancer` ou `normal`). L'objectif est d'extraire des caractéristiques visuelles, d'étudier les regroupements naturels des images, puis de comparer une approche supervisée à une approche semi-supervisée.

## Chiffres clés

| | Images |
|---|---:|
| Images brutes | 1 506 |
| dont annotées (50 `cancer`, 50 `normal`) | 100 |
| dont non annotées | 1 406 |
| **Après déduplication** | **1 387** |
| dont annotées (50 `cancer`, 48 `normal`) | 98 |

## Avancement

| Notebook | Contenu | Page |
|---|---|---|
| `notebooks/01.0-kl-data-exploration.ipynb` | Exploration initiale du jeu de données | [Exploration](donnees/exploration.md) |
| `notebooks/02.0-kl-duplicates-analysis.ipynb` | Analyse et règle de déduplication | [Doublons](donnees/doublons.md) |

Le code réutilisable vit dans `src/brain_mri/`. Les notebooks l'importent plutôt que de le recopier : voir la page [Préparation](pipeline/preparation.md) et la [référence de l'API](reference/dataset.md).

## Où trouver quoi

- [Démarrage](getting-started.md) : installer le projet et produire le jeu dédupliqué.
- [Données](donnees/jeu-de-donnees.md) : contenu du jeu de données, exploration et doublons.
- [Pipeline](pipeline/preparation.md) : flux de la préparation, fichiers produits.
- [Décisions](decisions.md) : choix faits et raisons.
