# OC-Appliquez-des-approches-semi-supervisees

Exploration et détection semi-supervisée de tumeurs cérébrales à partir d'IRM.

Le projet exploite des images IRM partiellement annotées afin d'extraire des
caractéristiques visuelles, d'étudier les regroupements naturels et de comparer
une approche supervisée à une approche semi-supervisée.

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/). Installez les dépendances :

```bash
uv sync
```

## Organisation du projet

```
├── Makefile           <- Commandes de développement courantes
├── README.md          <- Documentation principale du projet
├── data
│   ├── external       <- Données provenant de sources tierces
│   ├── interim        <- Données intermédiaires transformées
│   ├── processed      <- Données prêtes pour la modélisation
│   └── raw            <- Données brutes immuables
│
├── docs               <- Documentation MkDocs
│
├── models             <- Modèles entraînés et prédictions sérialisées
│
├── notebooks          <- Notebooks d'exploration et de modélisation
│
├── pyproject.toml     <- Métadonnées du package `brain-mri` et outils de développement
│
├── reports
│   └── figures        <- Graphiques destinés aux rapports et présentations
│
└── src
    └── brain_mri               <- Package Python du projet
        │
        ├── __init__.py         <- Initialise le package
        ├── config.py           <- Chemins et configuration partagée
        ├── dataset.py          <- Chargement et préparation des données
        ├── features.py         <- Extraction de caractéristiques visuelles
        ├── modeling
        │   ├── __init__.py
        │   ├── predict.py      <- Inférence avec les modèles entraînés
        │   └── train.py        <- Entraînement des modèles
        └── plots.py            <- Création des visualisations
```

## Commandes utiles

```bash
make lint    # Vérifier le formatage et le linting
make format  # Appliquer le formatage et les corrections Ruff
make test    # Exécuter les tests
```
