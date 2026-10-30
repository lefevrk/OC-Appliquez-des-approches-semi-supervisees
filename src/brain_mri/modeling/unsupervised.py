from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import NamedTuple

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import typer

from brain_mri.config import INTERIM_DATA_DIR, N_CLUSTERS, PROCESSED_DATA_DIR
from brain_mri.dataset import read_manifest, read_split
from brain_mri.features import MODEL_NAME, load_features
from brain_mri.modeling.preprocessing import (
    RANDOM_STATE,
    reduce_with_pca,
    standardize,
)

app = typer.Typer()

REFERENCE_LAYER = "layer4"
NOISE_LABEL = -1

# Codage des labels forts : 1 = cancer, 0 = normal. Les images non annotées
# n'ont pas de code : elles ne forment pas une classe.
CLASS_CODES = {"cancer": 1, "normal": 0}
CLASS_LABELS = ("cancer", "normal")
CLASS_ORDER = (CLASS_CODES["cancer"], CLASS_CODES["normal"])

CONFIDENCE_LEVEL = 0.95
_WILSON_Z = NormalDist().inv_cdf((1 + CONFIDENCE_LEVEL) / 2)


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Encode les labels forts en 1 (`cancer`) et 0 (`normal`).

    Args:
        labels: Labels `cancer`, `normal`, ou manquants pour les images non
            annotées.

    Returns:
        Un tableau de flottants, avec NaN pour les images non annotées.
    """
    return labels.map(CLASS_CODES).to_numpy(dtype=float)


def ari_without_noise(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> float:
    """Calcule l'ARI en ignorant les images classées comme bruit.

    Le bruit (`-1`) n'est ni un cluster ni une classe : DBSCAN et HDBSCAN y
    placent les images qu'aucune zone dense n'accepte.

    Args:
        cluster_labels: Cluster de chaque image (`-1` pour le bruit).
        true_labels: Label fort de chaque image.

    Returns:
        L'ARI sur les images hors bruit, ou NaN s'il en reste moins de deux ou
        si elles ne comptent qu'une classe.
    """
    kept = cluster_labels != NOISE_LABEL
    if kept.sum() < 2 or len(set(true_labels[kept])) < 2:
        return np.nan
    return float(adjusted_rand_score(true_labels[kept], cluster_labels[kept]))


def majority_mapping(
    cluster_labels: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> dict[int, int]:
    """Associe chaque cluster à la classe majoritaire de ses images annotées.

    En cas d'égalité, le cluster reçoit la classe `normal`. Un cluster sans
    image annotée n'a pas de classe. Le bruit n'est jamais associé à une
    classe.

    Args:
        cluster_labels: Cluster de chaque image.
        labels: Label fort de chaque image (1, 0 ou NaN).
        mask: Masque booléen des images annotées à utiliser pour le vote.

    Returns:
        Un dictionnaire cluster -> classe (1 = `cancer`, 0 = `normal`).
    """
    mapping = {}
    for cluster in sorted(set(cluster_labels) - {NOISE_LABEL}):
        members = mask & (cluster_labels == cluster)
        if members.any():
            mapping[cluster] = int(labels[members].mean() > 0.5)
    return mapping


def to_weak_labels(
    cluster_labels: np.ndarray, mapping: dict[int, int]
) -> np.ndarray:
    """Donne à chaque image la classe de son cluster.

    Args:
        cluster_labels: Cluster de chaque image.
        mapping: Dictionnaire cluster -> classe, issu de `majority_mapping`.

    Returns:
        Un tableau de flottants (1, 0), avec NaN pour une image dont le
        cluster n'a pas de classe.
    """
    return np.array(
        [mapping.get(cluster, np.nan) for cluster in cluster_labels]
    )


def kmeans_confidence(model: KMeans, space: np.ndarray) -> np.ndarray:
    """Mesure la marge de chaque image entre les deux centres de K-Means.

    La confiance vaut `(d2 - d1) / (d2 + d1)`, où `d1` est la distance au
    centre le plus proche et `d2` la distance à l'autre centre. Elle est nulle
    à égale distance des deux centres et vaut 1 sur le centre. C'est une marge
    géométrique, **pas une probabilité** d'avoir le bon label.

    Args:
        model: K-Means à deux clusters, déjà ajusté.
        space: Images à évaluer, dans l'espace où le modèle a été ajusté.

    Returns:
        La confiance de chaque image, entre 0 et 1.
    """
    distances = np.sort(model.transform(space), axis=1)
    return (distances[:, 1] - distances[:, 0]) / (
        distances[:, 1] + distances[:, 0]
    )


def build_weak_labels(
    images: pd.DataFrame,
    clusters: np.ndarray,
    mapping: dict[int, int],
    confidence: np.ndarray,
    method: str = "K-Means",
) -> pd.DataFrame:
    """Construit les labels faibles des images non annotées.

    Args:
        images: Images clusterisées, avec les colonnes `image_id` et
            `is_labeled`, dans l'ordre de `clusters`.
        clusters: Cluster de chaque image.
        mapping: Dictionnaire cluster -> classe, issu de `majority_mapping`.
        confidence: Confiance de chaque image.
        method: Nom de la méthode de clustering, recopié dans le résultat.

    Returns:
        Une ligne par image **non annotée** : `image_id`, `cluster`,
        `weak_label` (`cancer`, `normal`, ou vide si le cluster n'a pas de
        classe), `confidence` et `method`.
    """
    names = {code: name for name, code in CLASS_CODES.items()}
    weak = pd.DataFrame(
        {
            "image_id": images["image_id"],
            "cluster": clusters,
            "weak_label": pd.Series(
                to_weak_labels(clusters, mapping), index=images.index
            ).map(names),
            "confidence": confidence,
            "method": method,
        }
    )
    return weak[~images["is_labeled"].to_numpy()].reset_index(drop=True)


def wilson_interval(
    successes: int, total: int, z: float = _WILSON_Z
) -> tuple[float, float]:
    """Calcule l'intervalle de confiance de Wilson d'une proportion.

    Args:
        successes: Nombre de succès.
        total: Nombre d'essais.
        z: Quantile de la loi normale (par défaut, intervalle à 95 %).

    Returns:
        Les bornes basse et haute de l'intervalle.

    Raises:
        ValueError: Si `total` n'est pas strictement positif.
    """
    if total <= 0:
        raise ValueError("Le nombre d'essais doit être strictement positif.")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


class WeakLabelEvaluation(NamedTuple):
    """Résultat de `evaluate_weak_labels`.

    Attributes:
        confusion: Matrice de confusion (lignes : label fort, colonnes : label
            faible), `cancer` en premier.
        per_class: Précision, rappel, F1 et nombre d'images, par classe.
        summary: Nombre d'images, exactitude et son intervalle de confiance à
            95 %, exactitude équilibrée.
    """

    confusion: pd.DataFrame
    per_class: pd.DataFrame
    summary: dict[str, float]


def evaluate_weak_labels(
    expected: np.ndarray, predicted: np.ndarray
) -> WeakLabelEvaluation:
    """Évalue des labels faibles par rapport aux labels forts.

    Args:
        expected: Labels forts (1 = `cancer`, 0 = `normal`).
        predicted: Labels faibles correspondants, sans valeur manquante.

    Returns:
        La matrice de confusion, les mesures par classe et le résumé.
    """
    confusion = pd.DataFrame(
        confusion_matrix(expected, predicted, labels=list(CLASS_ORDER)),
        index=pd.Index(CLASS_LABELS, name="label fort"),
        columns=pd.Index(CLASS_LABELS, name="label faible"),
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=list(CLASS_ORDER), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "précision": precision,
            "rappel": recall,
            "F1": f1,
            "images": support,
        },
        index=pd.Index(CLASS_LABELS, name="classe"),
    )
    correct = int((expected == predicted).sum())
    low, high = wilson_interval(correct, len(expected))
    summary = {
        "images": len(expected),
        "exactitude": correct / len(expected),
        "IC 95 % bas": low,
        "IC 95 % haut": high,
        "exactitude équilibrée": float(
            balanced_accuracy_score(expected, predicted)
        ),
    }
    return WeakLabelEvaluation(confusion, per_class, summary)


@app.command()
def main(
    manifest_path: Path = INTERIM_DATA_DIR / "manifest.csv",
    split_path: Path = PROCESSED_DATA_DIR / "split.csv",
    features_dir: Path = PROCESSED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "weak_labels.csv",
    layer: str = REFERENCE_LAYER,
):
    """Labellise faiblement les images non annotées avec K-Means."""
    manifest = read_manifest(manifest_path)
    split = read_split(split_path).set_index("image_id")["split"]
    is_train = (split.loc[manifest["image_id"]] == "train").to_numpy()

    image_ids, features = load_features(
        features_dir / f"features_{MODEL_NAME}_{layer}.npz"
    )
    if not (image_ids == manifest["image_id"].to_numpy()).all():
        raise ValueError(
            "Les features ne sont pas alignées sur le manifeste : "
            "relancer `make features`."
        )

    reduced = reduce_with_pca(standardize(features, is_train), is_train)
    train_space = reduced[is_train]
    train_images = manifest[is_train].reset_index(drop=True)
    logger.info(
        f"K-Means sur {len(train_images)} images d'entraînement ({layer})..."
    )

    model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    model.fit(train_space)

    mapping = majority_mapping(
        model.labels_,
        encode_labels(train_images["label"]),
        train_images["is_labeled"].to_numpy(),
    )
    weak_labels = build_weak_labels(
        train_images,
        model.labels_,
        mapping,
        kmeans_confidence(model, train_space),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weak_labels.to_csv(output_path, index=False)
    logger.success(
        f"{len(weak_labels)} labels faibles enregistrés dans {output_path} "
        f"({weak_labels['weak_label'].value_counts().to_dict()})"
    )


if __name__ == "__main__":
    app()
