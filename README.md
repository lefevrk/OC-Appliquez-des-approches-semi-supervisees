<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">BrainScanAI - IRM cérébrales semi-supervisées</h3>

  <p align="center">
    Exploration et détection semi-supervisée de tumeurs cérébrales à partir d'IRM partiellement annotées : préparation des données, extraction de caractéristiques visuelles, regroupements naturels, puis comparaison d'une approche supervisée et d'une approche semi-supervisée.
    <br />
    <a href="https://lefevrk.github.io/OC-Appliquez-des-approches-semi-supervisees/"><strong>Explorer la doc »</strong></a>
    <br />
    <br />
    <a href="https://github.com/lefevrk/OC-Appliquez-des-approches-semi-supervisees/issues/new?labels=bug&template=bug_report.md">Signaler un bug</a>
    &middot;
    <a href="https://github.com/lefevrk/OC-Appliquez-des-approches-semi-supervisees/issues/new?labels=enhancement&template=feature_request.md">Proposer une fonctionnalité</a>
  </p>

  [![Docs](https://github.com/lefevrk/OC-Appliquez-des-approches-semi-supervisees/actions/workflows/docs.yml/badge.svg)](https://github.com/lefevrk/OC-Appliquez-des-approches-semi-supervisees/actions/workflows/docs.yml)
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Sommaire</summary>
  <ol>
    <li>
      <a href="#à-propos-du-projet">À propos du projet</a>
      <ul>
        <li><a href="#contexte">Contexte</a></li>
        <li><a href="#fonctionnalités">Fonctionnalités</a></li>
        <li><a href="#technologies-utilisées">Technologies utilisées</a></li>
        <li><a href="#feuille-de-route">Feuille de route</a></li>
      </ul>
    </li>
    <li>
      <a href="#démarrage">Démarrage</a>
      <ul>
        <li><a href="#prérequis">Prérequis</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#utilisation">Utilisation</a></li>
    <li><a href="#organisation-du-projet">Organisation du projet</a></li>
    <li><a href="#résultats-clés">Résultats clés</a></li>
    <li><a href="#remerciements">Remerciements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## À propos du projet

Ce dépôt est la première phase du projet fictif *BrainScanAI* de la startup e-santé *CurelyticsIA* : explorer la possibilité d'automatiser la détection de tumeurs du cerveau à partir d'un grand nombre d'IRM, dont **seules 100 sur 1 506 sont annotées** par des radiologues experts.

```mermaid
flowchart LR
    RAW[(IRM brutes<br/>1 506 images)] --> PREP[Nettoyage<br/>déduplication]
    PREP --> MAN[/Manifeste<br/>1 386 images/]
    MAN --> FEAT[Caractéristiques<br/>ResNet50 pré-entraîné]
    FEAT -.-> CLUS[Clustering<br/>labels faibles]
    CLUS -.-> SEMI[Apprentissage<br/>semi-supervisé]
```

Les flèches pointillées sont les étapes à venir.

### Contexte

Annoter une IRM demande l'avis d'un radiologue : c'est long et coûteux. Une approche semi-supervisée tire parti des nombreuses images non annotées pour compléter le faible signal des quelques labels disponibles. Le projet compare ainsi un modèle entraîné sur les seules images annotées à un modèle qui exploite aussi les images non annotées, puis formule des recommandations pour un passage à l'échelle.

### Fonctionnalités

- **Inventorie** les images brutes dans une table unique (identifiant, sous-ensemble, label, taille), sans jamais modifier `data/raw/`
- **Contrôle** la lisibilité, le format, les dimensions et le mode couleur de chaque image
- **Détecte les doublons exacts** (empreinte SHA-256) et les **quasi-doublons** (distance entre miniatures) pour éviter toute fuite entre apprentissage et évaluation
- **Applique une règle de déduplication** reproductible : la copie annotée est conservée en priorité, et **exclut** les images inexploitables
- **Produit un manifeste CSV** aux chemins relatifs, et un **rapport des suppressions** qui justifie chaque image retirée
- **Réduit les biais liés à l'image** par un prétraitement en niveaux de gris avec égalisation d'histogramme, dont l'effet est mesuré
- **Extrait des caractéristiques visuelles** avec un ResNet50 pré-entraîné et gelé (couches `layer3` et `layer4`), enregistrées avec leurs identifiants dans des fichiers `.npz`
- **Se teste et se documente** : tests `pytest` sur la logique de préparation, documentation MkDocs avec référence d'API générée

### Technologies utilisées

<ul>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=python" width="20" height="20" valign="middle" /> <strong>Python 3.12</strong> - langage principal</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=pandas" width="20" height="20" valign="middle" /> <strong>pandas</strong> - tables de données et manifeste</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=numpy" width="20" height="20" valign="middle" /> <strong>NumPy</strong> - calcul sur les images</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=sklearn" width="20" height="20" valign="middle" /> <strong>scikit-learn</strong> - recherche des plus proches voisins</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=matplotlib" width="20" height="20" valign="middle" /> <strong>Matplotlib</strong> - visualisations</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=jupyter" width="20" height="20" valign="middle" /> <strong>Jupyter</strong> - notebooks d'analyse</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=pytorch" width="20" height="20" valign="middle" /> <strong>PyTorch & torchvision</strong> - ResNet50 pré-entraîné et extraction des features</li>
  <li><img src="https://go-skill-icons.vercel.app/api/icons?i=pytest" width="20" height="20" valign="middle" /> <strong>pytest</strong> - tests</li>
</ul>

### Feuille de route

| Étape de la mission | État |
|---|---|
| 1. Importer les données et explorer le jeu de radiographies | Terminée |
| 2. Prétraiter et extraire les caractéristiques | Terminée |
| 3. Réaliser une analyse non supervisée | À venir |
| 4. Appliquer une méthode semi-supervisée | À venir |
| Synthèse et recommandations de passage à l'échelle | À venir |

<p align="right">(<a href="#readme-top">retour en haut</a>)</p>

<!-- GETTING STARTED -->
## Démarrage

### Prérequis

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) pour gérer les dépendances
- Le jeu de données d'IRM, à placer dans `data/raw/` (le dossier `data/` n'est pas versionné) :

  ```text
  data/raw/
  ├── sans_label/             # IRM sans annotation
  └── avec_labels/
      ├── cancer/             # IRM annotées « cancer »
      └── normal/             # IRM annotées « normal »
  ```

### Installation

1. Cloner le dépôt
   ```bash
   git clone https://github.com/lefevrk/OC-Appliquez-des-approches-semi-supervisees.git
   cd OC-Appliquez-des-approches-semi-supervisees
   ```

2. Installer les dépendances
   ```bash
   uv sync
   ```

<p align="right">(<a href="#readme-top">retour en haut</a>)</p>

<!-- USAGE EXAMPLES -->
## Utilisation

Détail complet dans la [documentation](docs/docs/getting-started.md).

### Produire le jeu dédupliqué

```bash
make data
```

La commande écrit `data/interim/manifest.csv` (images conservées) et `reports/removed_duplicates.csv` (images retirées, avec la raison). Ses options se listent avec `uv run python -m brain_mri.dataset --help`.

### Relire le manifeste

```python
from brain_mri.config import INTERIM_DATA_DIR
from brain_mri.dataset import read_manifest

manifest = read_manifest(INTERIM_DATA_DIR / "manifest.csv")
labeled = manifest.query("is_labeled")
```

### Extraire les features

```bash
make features
```

La commande lit le manifeste et écrit, dans `data/processed/`, `features_resnet50_layer3.npz` et `features_resnet50_layer4.npz` (identifiants et matrice de features). Les poids du ResNet50 sont téléchargés au premier lancement.

### Reproduire les analyses

| Notebook | Contenu |
|---|---|
| `notebooks/01.0-kl-data-exploration.ipynb` | Exploration initiale du jeu de données |
| `notebooks/02.0-kl-duplicates-analysis.ipynb` | Analyse et règle de déduplication |
| `notebooks/03.0-kl-feature-extraction.ipynb` | Qualité des images, prétraitement et extraction des features |

### Consulter la documentation

```bash
uv run mkdocs serve -f docs/mkdocs.yml
```

### Commandes utiles

```bash
make data      # Produire le manifeste dédupliqué et le rapport
make features  # Extraire les features du ResNet50
make lint      # Vérifier le formatage et le linting
make format    # Appliquer le formatage et les corrections Ruff
make test      # Exécuter les tests
```

<p align="right">(<a href="#readme-top">retour en haut</a>)</p>

## Organisation du projet

```
├── Makefile           <- Commandes de développement courantes
├── README.md          <- Documentation principale du projet
├── data
│   ├── external       <- Données provenant de sources tierces
│   ├── interim        <- Données intermédiaires transformées (manifeste dédupliqué)
│   ├── processed      <- Données prêtes pour la modélisation (features extraites)
│   └── raw            <- Données brutes immuables
│
├── docs               <- Documentation MkDocs (voir docs/mkdocs.yml)
│
├── models             <- Modèles entraînés et prédictions sérialisées
│
├── notebooks          <- Notebooks d'exploration et de modélisation
│
├── pyproject.toml     <- Métadonnées du package `brain-mri` et outils de développement
│
├── references         <- Énoncé de la mission et livrables attendus
│
├── reports
│   ├── figures        <- Graphiques destinés aux rapports et présentations
│   └── removed_duplicates.csv  <- Images retirées par la déduplication
│
├── src
│   └── brain_mri               <- Package Python du projet
│       │
│       ├── __init__.py         <- Initialise le package
│       ├── config.py           <- Chemins et configuration partagée
│       ├── dataset.py          <- Chargement, déduplication et manifeste des données
│       ├── features.py         <- Extraction de caractéristiques visuelles
│       ├── modeling
│       │   ├── __init__.py
│       │   ├── predict.py      <- Inférence avec les modèles entraînés
│       │   └── train.py        <- Entraînement des modèles
│       └── plots.py            <- Création des visualisations
│
└── tests              <- Tests unitaires
```

<p align="right">(<a href="#readme-top">retour en haut</a>)</p>

## Résultats clés

- Les **1 506 images** (100 annotées, 1 406 non annotées) sont toutes lisibles, en JPEG, RGB et 512 × 512 - détail dans [Exploration](docs/docs/donnees/exploration.md).
- **90 groupes de doublons exacts** (186 fichiers) et **24 paires de quasi-doublons** ont été trouvés, sans aucun label contradictoire - détail dans [Doublons](docs/docs/donnees/doublons.md).
- Après déduplication et exclusion d'une image inexploitable, **1 386 images** sont conservées, dont **98 annotées** (50 `cancer`, 48 `normal`) - détail dans [Préparation](docs/docs/pipeline/preparation.md).
- Sur les images d'origine, la luminosité, le contraste et la part de fond suffisent à séparer en grande partie les classes annotées (AUC de 0,82). Après égalisation d'histogramme, cet indice tombe à 0,66 sans dégrader les features - détail dans [Extraction des features](docs/docs/pipeline/features.md).
- Les features du ResNet50 portent un signal : un modèle linéaire de référence atteint une AUC de 0,978 (`layer3`) et 0,966 (`layer4`) sur les 98 images annotées, en validation croisée.
- **34 tests** couvrent l'inventaire, la déduplication, le manifeste, le prétraitement et l'extraction.

<!-- ACKNOWLEDGMENTS -->
## Remerciements

Projet réalisé par **Kilian LEFEVRE** dans le cadre du parcours OpenClassrooms AI Engineer - Projet : *Labellisez et appliquez des approches semi-supervisées en traitement d'images*.

<p align="right">(<a href="#readme-top">retour en haut</a>)</p>
