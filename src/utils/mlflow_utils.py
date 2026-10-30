"""Outils de suivi d'expériences MLflow, en local.

Les runs sont écrits dans une base SQLite (`mlflow.db`) et les artefacts dans
le dossier `mlruns/`, tous deux à la racine du projet et ignorés par Git. Une
seule expérience (`EXPERIMENT_NAME`) regroupe tous les runs : la phase et le
notebook sont des **tags** qui servent de filtres dans l'interface
(`make mlflow-ui`).

Un run correspond à **une configuration testée**, pas à un notebook.

Variables d'environnement :
    MLFLOW_TRACKING_URI: remplace la base SQLite locale (par exemple pour un
        serveur de suivi).
    MLFLOW_UI_URL: adresse de l'interface, utilisée pour les liens affichés
        dans les notebooks (par défaut `http://127.0.0.1:5001`).
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any
import warnings

import mlflow
import mlflow.data
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import sklearn
import torch

from brain_mri.config import PROJ_ROOT, SEED, mask_path

EXPERIMENT_NAME = "BrainScanAI"
DEFAULT_DATABASE = PROJ_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJ_ROOT / "mlruns"
DEFAULT_UI_URL = "http://127.0.0.1:5001"
DESCRIPTION_TAG = "mlflow.note.content"
UNKNOWN = "unknown"


# --- Configuration -----------------------------------------------------------


def get_tracking_uri(tracking_uri: str | None = None) -> str:
    """Retourne l'adresse de la base de suivi.

    Ordre de priorité : l'argument, la variable `MLFLOW_TRACKING_URI`, puis la
    base SQLite locale `mlflow.db`.

    Args:
        tracking_uri: Adresse explicite, ou `None`.

    Returns:
        L'adresse de suivi à utiliser.
    """
    return (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or f"sqlite:///{DEFAULT_DATABASE}"
    )


def setup_mlflow(
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> str:
    """Configure MLflow et active l'expérience du projet.

    L'expérience est créée à la première utilisation, avec ses artefacts dans
    `artifacts_dir` (et non dans le dossier courant).

    Args:
        experiment_name: Nom de l'expérience.
        tracking_uri: Adresse de la base de suivi (voir `get_tracking_uri`).
        artifacts_dir: Dossier des artefacts de l'expérience, à sa création.

    Returns:
        L'adresse de suivi masquée (sans chemin local), prête à afficher.
    """
    # Les messages d'information et les avertissements de MLflow n'apportent
    # rien dans un notebook : seules les erreurs sont affichées.
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(get_tracking_uri(tracking_uri))
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment_name, artifact_location=artifacts_dir.as_uri()
        )
    mlflow.set_experiment(experiment_name)
    return mask_path(mlflow.get_tracking_uri())


# --- Contexte d'exécution ----------------------------------------------------


def _git(*arguments: str) -> str:
    """Exécute une commande Git à la racine du projet, `unknown` si elle échoue."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier, `unknown` s'il est absent."""
    if not path.is_file():
        return UNKNOWN
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_device_name() -> str:
    """Nom du calcul disponible : `cuda`, `mps` (Apple) ou `cpu`."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def runtime_tags() -> dict[str, str]:
    """Décrit l'environnement d'exécution (versions, calcul).

    Returns:
        Un dictionnaire de tags : Python, système, versions des bibliothèques
        et appareil de calcul.
    """
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
        "device": select_device_name(),
    }


def provenance_tags(manifest_path: Path | None = None) -> dict[str, str]:
    """Relie un run au code et aux données qui l'ont produit.

    Args:
        manifest_path: Manifeste du jeu de données, dont l'empreinte identifie
            la version des données (déduplication comprise).

    Returns:
        Les tags `git_commit`, `git_dirty`, `uv_lock_sha256` et, si un
        manifeste est donné, `manifest_sha256`.
    """
    tags = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))).lower(),
        "uv_lock_sha256": file_sha256(PROJ_ROOT / "uv.lock"),
    }
    if manifest_path is not None:
        tags["manifest_sha256"] = file_sha256(manifest_path)
    return tags


# --- Masquage des sorties ----------------------------------------------------


@contextmanager
def mask_mlflow_output() -> Iterator[None]:
    """Capture la sortie de MLflow et masque les chemins locaux avant de l'afficher."""
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        for text in (stdout.getvalue(), stderr.getvalue()):
            if text:
                print(mask_path(text), end="")


# --- Runs --------------------------------------------------------------------


@contextmanager
def start_run(
    run_name: str,
    phase: str,
    notebook: str,
    description: str | None = None,
    manifest_path: Path | None = None,
    seed: int | None = SEED,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Démarre un run à plat, avec les tags de provenance et d'environnement.

    Args:
        run_name: Nom lisible de la configuration (`kmeans-k3-layer3`).
        phase: Phase du projet (`features`, `unsupervised`...), en tag.
        notebook: Notebook qui crée le run, en tag.
        description: Description du run, affichée dans l'interface.
        manifest_path: Manifeste du jeu de données (empreinte en tag).
        seed: Graine utilisée, en tag (`None` si sans objet).
        tags: Tags supplémentaires.

    Yields:
        Le run actif.
    """
    all_tags: dict[str, Any] = {
        "phase": phase,
        "notebook": notebook,
        **provenance_tags(manifest_path),
        **runtime_tags(),
        **(tags or {}),
    }
    if seed is not None:
        all_tags["seed"] = seed
    if description:
        all_tags[DESCRIPTION_TAG] = description
    # Seuls les messages de MLflow sont masqués : les affichages du notebook
    # dans le run ne sont ni capturés ni retardés.
    with mask_mlflow_output():
        run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags({key: str(value) for key, value in all_tags.items()})
    try:
        yield run
    finally:
        with mask_mlflow_output():
            mlflow.end_run()


@contextmanager
def resume_run(run_id: str) -> Iterator[mlflow.ActiveRun]:
    """Reprend un run terminé pour lui ajouter des métriques ou des figures.

    Args:
        run_id: Identifiant du run à reprendre.

    Yields:
        Le run actif.
    """
    with mask_mlflow_output():
        run = mlflow.start_run(run_id=run_id)
    try:
        yield run
    finally:
        with mask_mlflow_output():
            mlflow.end_run()


# --- Journalisation ----------------------------------------------------------


def _to_param_value(value: Any) -> Any:
    """Convertit une valeur en type accepté par `mlflow.log_params`."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def log_params(params: Mapping[str, Any]) -> None:
    """Enregistre les paramètres, convertis en types simples."""
    mlflow.log_params({key: _to_param_value(v) for key, v in params.items()})


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Enregistre les métriques, en ignorant les valeurs absentes ou non finies.

    Args:
        metrics: Nom de la métrique -> valeur.
        step: Étape (époque, valeur de k...) pour tracer une courbe.
    """
    clean = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    }
    mlflow.log_metrics(clean, step=step)


def log_figure(figure: Any, name: str) -> None:
    """Enregistre une figure Matplotlib comme artefact du run.

    Args:
        figure: Figure Matplotlib.
        name: Nom du fichier image (`pca-features.png`).
    """
    mlflow.log_figure(figure, name)


def log_dataset(
    dataframe: pd.DataFrame,
    name: str,
    context: str,
    source: str,
    targets: str | None = None,
) -> None:
    """Enregistre un jeu de données comme entrée du run (schéma MLflow).

    MLflow conserve le profil, le schéma et l'empreinte du jeu : ni le tableau
    ni le fichier ne sont copiés en artefact.

    Args:
        dataframe: Table décrivant le jeu.
        name: Nom du jeu (`manifest`, `split-train`...).
        context: Rôle du jeu dans le run (`training`, `validation`, `test`).
        source: Chemin du fichier d'origine, **relatif au projet**.
        targets: Colonne des labels, si elle existe.
    """
    with warnings.catch_warnings():
        # Un chemin relatif peut désigner plusieurs types de sources : MLflow
        # choisit une source locale, ce qui convient ici.
        warnings.filterwarnings(
            "ignore", message="The specified dataset source can be interpreted"
        )
        dataset = mlflow.data.from_pandas(
            dataframe, source=source, name=name, targets=targets
        )
    mlflow.log_input(dataset, context=context)


def log_sklearn_model(model: Any, name: str) -> None:
    """Enregistre un modèle scikit-learn ajusté (scaler, PCA, K-Means...)."""
    mlflow.sklearn.log_model(sk_model=model, name=name)


def log_pytorch_model(model: torch.nn.Module, name: str) -> None:
    """Enregistre un modèle PyTorch."""
    mlflow.pytorch.log_model(pytorch_model=model, name=name)


# --- Lecture -----------------------------------------------------------------


def search_runs_df(
    experiment_names: list[str] | None = None,
    filter_string: str = "",
    order_by: list[str] | None = None,
) -> pd.DataFrame:
    """Lit les runs dans un tableau.

    Args:
        experiment_names: Expériences à lire (celle du projet par défaut).
        filter_string: Filtre MLflow, par exemple `tags.phase = 'unsupervised'`.
        order_by: Critères de tri, par exemple `["metrics.ari DESC"]`.

    Returns:
        Une ligne par run.
    """
    return mlflow.search_runs(
        experiment_names=experiment_names or [EXPERIMENT_NAME],
        filter_string=filter_string,
        order_by=order_by,
    )


def get_best_run(
    metric: str,
    higher_is_better: bool = True,
    filter_string: str = "",
    experiment_names: list[str] | None = None,
) -> pd.Series:
    """Retourne le meilleur run selon une métrique.

    Args:
        metric: Nom de la métrique.
        higher_is_better: `False` pour une perte ou une erreur.
        filter_string: Filtre MLflow (par exemple sur la phase).
        experiment_names: Expériences à lire.

    Returns:
        La ligne du meilleur run.

    Raises:
        ValueError: Si aucun run ne correspond.
    """
    direction = "DESC" if higher_is_better else "ASC"
    runs = search_runs_df(
        experiment_names, filter_string, [f"metrics.{metric} {direction}"]
    )
    if runs.empty:
        raise ValueError("Aucun run ne correspond à la recherche.")
    return runs.iloc[0]


# --- Affichage ---------------------------------------------------------------


def display_run_link(run: mlflow.ActiveRun | mlflow.entities.Run) -> None:
    """Affiche dans le notebook le lien vers un run, sans chemin local."""
    from IPython.display import Markdown, display

    base = os.getenv("MLFLOW_UI_URL", DEFAULT_UI_URL).rstrip("/")
    experiment_id = run.info.experiment_id
    run_id = run.info.run_id
    display(
        Markdown(
            f"[Ouvrir le run `{run.info.run_name}`]"
            f"({base}/#/experiments/{experiment_id}/runs/{run_id})"
        )
    )
