# Module `utils.mlflow_utils`

Suivi des expériences avec MLflow, **en local**. Le module vit dans `src/utils/`, à côté de `src/brain_mri/` : ce sont les notebooks qui l'appellent, les modules de `brain_mri` ne dépendent pas de MLflow. Les principes de suivi sont décrits dans la page [Conventions](../conventions.md#suivi-des-experiences).

## Où sont les runs

- Les runs sont écrits dans une base SQLite, `mlflow.db`, et les artefacts dans `mlruns/`, tous deux à la racine du projet et **ignorés par git**.
- La variable d'environnement `MLFLOW_TRACKING_URI` remplace la base locale, par exemple pour un serveur de suivi.
- L'interface s'ouvre avec :

```bash
make mlflow-ui
```

Elle est alors disponible à l'adresse `http://127.0.0.1:5001` (`MLFLOW_UI_URL` pour la changer, dans les liens affichés par les notebooks).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `EXPERIMENT_NAME` | `"BrainScanAI"` | Expérience unique du projet |
| `DEFAULT_DATABASE` | `mlflow.db` | Base de suivi locale |
| `ARTIFACTS_DIR` | `mlruns/` | Dossier des artefacts |
| `DEFAULT_UI_URL` | `http://127.0.0.1:5001` | Adresse de l'interface |

## Configuration

::: utils.mlflow_utils.get_tracking_uri

::: utils.mlflow_utils.setup_mlflow

## Contexte d'exécution

Chaque run reçoit des **tags** qui le relient au code, aux données et à l'environnement qui l'ont produit : `git_commit`, `git_dirty`, `uv_lock_sha256`, `manifest_sha256`, `seed`, versions de Python et des bibliothèques, appareil de calcul, et `phase` et `notebook` pour filtrer dans l'interface.

::: utils.mlflow_utils.runtime_tags

::: utils.mlflow_utils.provenance_tags

::: utils.mlflow_utils.mask_mlflow_output

## Runs

::: utils.mlflow_utils.start_run

::: utils.mlflow_utils.resume_run

## Journalisation

Un jeu de données est enregistré avec le schéma MLflow (`log_dataset`) : MLflow garde son profil, son schéma et son empreinte, sans copier le tableau ni le fichier en artefact. Les artefacts sont uniquement des figures et des modèles ajustés.

::: utils.mlflow_utils.log_params

::: utils.mlflow_utils.log_metrics

::: utils.mlflow_utils.log_figure

::: utils.mlflow_utils.log_dataset

::: utils.mlflow_utils.log_sklearn_model

::: utils.mlflow_utils.log_pytorch_model

## Consultation

::: utils.mlflow_utils.search_runs_df

::: utils.mlflow_utils.get_best_run

::: utils.mlflow_utils.display_run_link
