# Conventions

Les règles de travail du projet : nommage des notebooks, organisation du code, suivi des expériences et workflow git.

## Notebooks

Les notebooks se nomment `PHASE.NUMÉRO-initiales-description` (standard cookiecutter-data-science).

| Phase | Contenu | Notebooks |
|---|---|---|
| `01` | Exploration des données | `01.01-kl-data-exploration`, `01.02-kl-image-quality` |
| `02` | Préparation | `02.01-kl-data-cleaning`, `02.02-kl-feature-extraction` |
| `03` | Modélisation | `03.01-kl-unsupervised-analysis` |

Règles communes :

- Un notebook **raconte une analyse** : il pose des questions, mesure, conclut. Son texte se lit seul.
- La **logique réutilisable vit dans `src/`** et est testée par `pytest` ; les notebooks l'importent.
- Les sorties ne contiennent **aucun chemin personnel** : voir [`brain_mri.config`](reference/config.md).
- Les notebooks sont **modifiés sur place**, jamais régénérés par un script.
- Les jeux d'entraînement, de validation et de test sont définis **avant toute transformation** : rien ne s'ajuste sur le test.

## Code

- Le code du projet est dans `src/brain_mri/`. Le suivi des expériences, appelé uniquement par les notebooks, est dans `src/utils/`.
- Docstrings au format Google, en français ; la documentation de référence est générée à partir d'elles.
- Dans `src/`, une condition invalide **lève une exception explicite** : `assert` est réservé aux notebooks et aux tests.
- Avant chaque commit, `pre-commit` vérifie le formatage (`ruff`), le linting et lance les tests.

```bash
make lint    # Vérifier le formatage et le linting
make format  # Appliquer le formatage et les corrections
make test    # Exécuter les tests
```

## Suivi des expériences

Le suivi utilise MLflow, en local (voir [`utils.mlflow_utils`](reference/tracking.md)).

- **Une seule expérience** (`BrainScanAI`) pour tout le projet. La phase et le notebook sont des **tags** qui servent de filtres dans l'interface.
- **Un run correspond à une configuration testée**, pas à un notebook : une valeur de `k`, une configuration de DBSCAN, un seuil de décision. Les runs sont à plat, sans run parent.
- Seuls les notebooks qui **comparent des configurations** créent des runs : aujourd'hui `03.01-kl-unsupervised-analysis`. Les notebooks d'exploration et de nettoyage prennent des décisions, qui vivent dans le manifeste, le rapport des suppressions et la page [Décisions](decisions.md).
- Chaque run enregistre ses paramètres, ses métriques, le jeu d'entraînement (schéma MLflow, sans copie du fichier) et, comme artefacts, des figures et les modèles ajustés.
- Les mesures du **test** sont consignées pour mémoire : on ne s'en sert ni pour comparer des runs, ni pour en choisir un.

```bash
make mlflow-ui   # Ouvrir l'interface sur la base locale
```

## Workflow git

Le dépôt suit un **git-flow** :

| Branche | Rôle |
|---|---|
| `main` | Version publiée. La documentation est construite depuis cette branche |
| `develop` | Intégration des travaux terminés |
| `feature/...` | Un chantier, créé depuis `develop` |
| `release/x.y.z` | Préparation d'une version, fusionnée dans `main` et `develop` |

Les messages de commit suivent le format `type(portée): message`, en anglais, à l'impératif : `feat(notebooks)`, `fix`, `refactor`, `test`, `docs`, `build`, `chore`.

Dans une branche, l'ordre habituel est : le notebook d'abord, puis le code extrait dans `src/`, puis les tests, puis la documentation. Celle-ci arrive en dernier, une fois les chiffres des notebooks stabilisés, parce que le site est publié depuis `main`.
