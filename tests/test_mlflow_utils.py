import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from brain_mri.config import PROJ_ROOT, SAFE_ROOT_LABEL
from utils import mlflow_utils
from utils.mlflow_utils import (
    EXPERIMENT_NAME,
    file_sha256,
    get_best_run,
    get_tracking_uri,
    log_dataset,
    log_figure,
    log_metrics,
    log_params,
    log_sklearn_model,
    mask_mlflow_output,
    resume_run,
    search_runs_df,
    setup_mlflow,
    start_run,
)


@pytest.fixture
def tracking(tmp_path: Path):
    """Base de suivi et artefacts temporaires, avec l'état MLflow restauré."""
    previous_uri = mlflow.get_tracking_uri()
    setup_mlflow(
        tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        artifacts_dir=tmp_path / "artifacts",
    )
    yield tmp_path
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def _run(name: str = "essai", **kwargs):
    return start_run(name, phase="tests", notebook="test", **kwargs)


def test_get_tracking_uri_defaults_to_the_local_sqlite_database(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert get_tracking_uri() == f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"


def test_get_tracking_uri_prefers_argument_then_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://exemple:5000")

    assert get_tracking_uri() == "http://exemple:5000"
    assert get_tracking_uri("sqlite:///autre.db") == "sqlite:///autre.db"


def test_setup_mlflow_creates_one_experiment_with_local_artifacts(
    tracking: Path,
):
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    assert experiment is not None
    assert experiment.artifact_location == (tracking / "artifacts").as_uri()


def test_setup_mlflow_twice_keeps_a_single_experiment(tracking: Path):
    setup_mlflow(
        tracking_uri=f"sqlite:///{tracking / 'test.db'}",
        artifacts_dir=tracking / "artifacts",
    )

    names = [e.name for e in mlflow.search_experiments()]

    assert names.count(EXPERIMENT_NAME) == 1


def test_setup_mlflow_returns_a_masked_uri(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_utils, "DEFAULT_DATABASE", PROJ_ROOT / "x.db")
    previous_uri = mlflow.get_tracking_uri()
    try:
        masked = setup_mlflow(
            tracking_uri=f"sqlite:///{PROJ_ROOT / 'tmp-test.db'}",
            artifacts_dir=tmp_path / "artifacts",
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)
        (PROJ_ROOT / "tmp-test.db").unlink(missing_ok=True)

    assert str(PROJ_ROOT) not in masked
    assert SAFE_ROOT_LABEL in masked


def test_start_run_sets_phase_notebook_and_provenance_tags(tracking: Path):
    manifest = tracking / "manifest.csv"
    manifest.write_text("image_id\na\n")

    with _run("kmeans-k2", manifest_path=manifest, seed=3) as run:
        pass

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["phase"] == "tests"
    assert tags["notebook"] == "test"
    assert tags["seed"] == "3"
    assert tags["manifest_sha256"] == file_sha256(manifest)
    assert {"git_commit", "git_dirty", "uv_lock_sha256"} <= set(tags)
    assert {"python_version", "torch_version", "device"} <= set(tags)


def test_start_run_stores_the_description(tracking: Path):
    with _run(description="Balayage de k") as run:
        pass

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["mlflow.note.content"] == "Balayage de k"


def test_start_run_ends_the_run_even_when_the_body_fails(tracking: Path):
    with pytest.raises(RuntimeError):
        with _run():
            raise RuntimeError("échec")

    assert mlflow.active_run() is None


def test_start_run_does_not_delay_the_output_of_the_body(
    tracking: Path, capsys
):
    with _run():
        print("pendant le run")
        during = capsys.readouterr().out

    assert "pendant le run" in during


def test_runs_are_flat_and_in_a_single_experiment(tracking: Path):
    for name in ("a", "b", "c"):
        with _run(name):
            pass

    runs = search_runs_df()

    assert len(runs) == 3
    assert (
        "tags.mlflow.parentRunId" not in runs
        or runs["tags.mlflow.parentRunId"].isna().all()
    )


def test_log_params_converts_values_to_simple_types(tracking: Path):
    with _run() as run:
        log_params({"k": 3, "layers": ("layer3", "layer4"), "seed": None})

    params = mlflow.get_run(run.info.run_id).data.params
    assert params["k"] == "3"
    assert params["layers"] == "('layer3', 'layer4')"


def test_log_metrics_skips_missing_and_non_finite_values(tracking: Path):
    with _run() as run:
        log_metrics({"ari": 0.2, "silhouette": float("nan"), "noise": None})

    metrics = mlflow.get_run(run.info.run_id).data.metrics
    assert metrics == {"ari": 0.2}


def test_log_metrics_records_the_step(tracking: Path):
    with _run() as run:
        for step, value in enumerate([3.0, 2.0, 1.5]):
            log_metrics({"inertia": value}, step=step)

    history = mlflow.MlflowClient().get_metric_history(
        run.info.run_id, "inertia"
    )
    assert [m.step for m in history] == [0, 1, 2]


def test_log_dataset_records_the_dataset_without_copying_it(tracking: Path):
    table = pd.DataFrame(
        {"image_id": ["a", "b"], "label": ["cancer", "normal"]}
    )

    with _run() as run:
        log_dataset(
            table,
            name="manifest",
            context="training",
            source="data/interim/manifest.csv",
            targets="label",
        )

    stored = mlflow.get_run(run.info.run_id)
    (dataset_input,) = stored.inputs.dataset_inputs
    assert dataset_input.dataset.name == "manifest"
    assert dataset_input.dataset.source == (
        '{"uri": "data/interim/manifest.csv"}'
    )
    tags = {t.key: t.value for t in dataset_input.tags}
    assert tags["mlflow.data.context"] == "training"
    assert mlflow.MlflowClient().list_artifacts(run.info.run_id) == []


def test_log_figure_stores_an_image_artifact(tracking: Path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    with _run() as run:
        log_figure(figure, "courbe.png")
    plt.close(figure)

    names = [
        a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)
    ]
    assert names == ["courbe.png"]


def test_log_sklearn_model_can_be_reloaded(tracking: Path):
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(points)

    with _run() as run:
        log_sklearn_model(model, "kmeans")

    reloaded = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/kmeans")
    assert (reloaded.predict(points) == model.predict(points)).all()


def test_get_best_run_follows_the_metric_direction(tracking: Path):
    for name, loss in (("a", 0.9), ("b", 0.2), ("c", 0.5)):
        with _run(name):
            log_metrics({"loss": loss})

    lowest = get_best_run("loss", higher_is_better=False)
    highest = get_best_run("loss", higher_is_better=True)

    assert lowest["tags.mlflow.runName"] == "b"
    assert highest["tags.mlflow.runName"] == "a"


def test_get_best_run_raises_when_there_is_no_run(tracking: Path):
    with pytest.raises(ValueError):
        get_best_run("loss")


def test_mask_mlflow_output_hides_local_paths(capsys):
    with mask_mlflow_output():
        print(f"lecture de {PROJ_ROOT}/mlruns/1")

    printed = capsys.readouterr().out
    assert str(PROJ_ROOT) not in printed
    assert f"{SAFE_ROOT_LABEL}/mlruns/1" in printed


def test_file_sha256_returns_unknown_for_a_missing_file(tmp_path: Path):
    assert file_sha256(tmp_path / "absent.csv") == "unknown"


def test_resume_run_adds_metrics_to_a_finished_run(tracking: Path):
    with _run("kmeans") as run:
        log_metrics({"ari": 0.2})

    with resume_run(run.info.run_id):
        log_metrics({"accuracy": 0.8})

    metrics = mlflow.get_run(run.info.run_id).data.metrics
    assert metrics == {"ari": 0.2, "accuracy": 0.8}
    assert mlflow.active_run() is None


def test_resume_run_ends_the_run_even_when_the_body_fails(tracking: Path):
    with _run("kmeans") as run:
        pass

    with pytest.raises(RuntimeError):
        with resume_run(run.info.run_id):
            raise RuntimeError("échec")

    assert mlflow.active_run() is None


def test_setup_mlflow_keeps_only_the_errors_of_mlflow(tracking: Path):
    assert logging.getLogger("mlflow").level == logging.ERROR
