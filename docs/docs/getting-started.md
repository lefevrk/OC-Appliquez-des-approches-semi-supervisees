# Démarrage

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

## Données attendues

Les images brutes se placent dans `data/raw/`, avec l'organisation suivante :

```text
data/raw/
├── sans_label/             # IRM sans annotation
└── avec_labels/
    ├── cancer/             # IRM annotées « cancer »
    └── normal/             # IRM annotées « normal »
```

!!! note
    Le dossier `data/` est ignoré par git : les données doivent être copiées manuellement. `data/raw/` est **immuable**, aucune étape ne le modifie.

## Produire le jeu dédupliqué

```bash
make data
```

La commande lance `brain_mri.dataset` et écrit :

| Fichier | Contenu |
|---|---|
| `data/interim/manifest.csv` | Les images conservées, avec leur label et leur empreinte |
| `reports/removed_duplicates.csv` | Les images retirées, avec la raison et l'image conservée |

Les options se règlent en ligne de commande :

```bash
uv run python -m brain_mri.dataset --help
```

## Extraire les features

```bash
make features
```

La commande lit `data/interim/manifest.csv` et écrit, dans `data/processed/`, un fichier par couche du ResNet50 : `features_resnet50_layer3.npz` et `features_resnet50_layer4.npz`. Les poids du modèle sont téléchargés au premier lancement. Les options se listent avec `--help`.

!!! note
    `uv sync` installe PyTorch, dont le téléchargement est volumineux.

## Relire le manifeste

```python
from PIL import Image

from brain_mri.config import INTERIM_DATA_DIR
from brain_mri.dataset import read_manifest

manifest = read_manifest(INTERIM_DATA_DIR / "manifest.csv")
labeled = manifest.query("is_labeled")

with Image.open(manifest.loc[0, "filepath"]) as image:
    ...
```

## Commandes utiles

```bash
make lint    # Vérifier le formatage et le linting
make format  # Appliquer le formatage et les corrections Ruff
make test    # Exécuter les tests
```

## Construire cette documentation

```bash
uv run mkdocs serve -f docs/mkdocs.yml   # aperçu local
uv run mkdocs build -f docs/mkdocs.yml   # génération du site, dans docs/site
```
