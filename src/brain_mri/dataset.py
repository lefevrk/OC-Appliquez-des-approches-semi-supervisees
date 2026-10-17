from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import NamedTuple

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import typer

from brain_mri.config import INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

IMAGE_PATTERN = "*.jpg"

# Détection des quasi-doublons : seuil choisi par inspection visuelle des
# paires dans `notebooks/02.0-kl-duplicates-analysis.ipynb`.
THUMBNAIL_SIZE = 32
NEAR_DUPLICATE_THRESHOLD = 0.15

# Images retirées à la main, car inexploitables (cf. notebook 03.0). Le fichier
# brut n'est jamais modifié : l'image est seulement exclue du manifeste.
EXCLUDED_IMAGES: dict[str, str] = {
    "ca4c389b-8e9b-40d0-b149-1c81934ea251": (
        "saturée, sans détail anatomique exploitable"
    ),
}

# Sous-dossier de `data/raw/` et label associé (NA = image non annotée).
DATA_SOURCES: dict[str, tuple[Path, str | None]] = {
    "sans_label": (Path("sans_label"), None),
    "cancer": (Path("avec_labels") / "cancer", "cancer"),
    "normal": (Path("avec_labels") / "normal", "normal"),
}

REMOVED_COLUMNS = [
    "image_id",
    "subset",
    "kept_image_id",
    "kept_subset",
    "reason",
    "distance",
]

INVENTORY_COLUMNS = [
    "image_id",
    "filepath",
    "subset",
    "label",
    "is_labeled",
    "file_size_bytes",
]


def build_inventory(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Construit la table de référence des images, une ligne par fichier.

    Args:
        raw_dir: Répertoire racine des données brutes.

    Returns:
        Un DataFrame trié par `subset` puis `image_id`. La colonne `label`
            vaut `<NA>` pour les images non annotées.

    Raises:
        FileNotFoundError: Si un des sous-dossiers attendus est absent.
    """
    records = []
    for subset, (relative_dir, label) in DATA_SOURCES.items():
        source_dir = raw_dir / relative_dir
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Répertoire introuvable : {source_dir}")
        records.extend(
            {
                "image_id": path.stem,
                "filepath": path,
                "subset": subset,
                "label": label if label is not None else pd.NA,
                "is_labeled": label is not None,
                "file_size_bytes": path.stat().st_size,
            }
            for path in source_dir.glob(IMAGE_PATTERN)
        )

    return (
        pd.DataFrame(records, columns=INVENTORY_COLUMNS)
        .sort_values(["subset", "image_id"])
        .reset_index(drop=True)
    )


def file_hash(filepath: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Args:
        filepath: Chemin du fichier à lire.

    Returns:
        L'empreinte hexadécimale du contenu binaire du fichier.
    """
    with filepath.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def add_file_hash(inventory: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `sha256`, utile pour repérer les doublons exacts.

    Args:
        inventory: Table produite par `build_inventory`.

    Returns:
        Une copie de `inventory` avec la colonne `sha256`.
    """
    return inventory.assign(sha256=inventory["filepath"].map(file_hash))


def deduplicate_exact(inventory: pd.DataFrame) -> pd.DataFrame:
    """Conserve une seule copie par contenu binaire identique.

    Dans chaque groupe de même `sha256`, la copie annotée est conservée en
    priorité, sinon celle dont l'`image_id` est le plus petit.

    Args:
        inventory: Table produite par `add_file_hash`.

    Returns:
        L'inventaire sans doublons exacts, trié par `subset` puis `image_id`.
    """
    return (
        inventory.sort_values(
            ["sha256", "is_labeled", "image_id"], ascending=[True, False, True]
        )
        .drop_duplicates("sha256", keep="first")
        .sort_values(["subset", "image_id"])
        .reset_index(drop=True)
    )


def load_thumbnail(filepath: Path, size: int = THUMBNAIL_SIZE) -> np.ndarray:
    """Charge une miniature en niveaux de gris, aplatie, valeurs dans [0, 1].

    Args:
        filepath: Chemin de l'image.
        size: Côté de la miniature carrée, en pixels.

    Returns:
        Un vecteur de `size * size` valeurs.
    """
    with Image.open(filepath) as image:
        thumbnail = image.convert("L").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    return np.asarray(thumbnail, dtype=np.float32).ravel() / 255


def find_near_duplicate_pairs(
    inventory: pd.DataFrame,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Repère les paires d'images dont les miniatures sont quasi identiques.

    Chaque image est comparée à son plus proche voisin (distance euclidienne
    entre miniatures). Les doublons exacts doivent être retirés au préalable.

    Args:
        inventory: Table avec les colonnes `image_id` et `filepath`.
        threshold: Distance en dessous de laquelle une paire est retenue.

    Returns:
        Un DataFrame `image_id_a`, `image_id_b`, `distance`, avec
            `image_id_a < image_id_b`, trié par distance croissante.
    """
    columns = ["image_id_a", "image_id_b", "distance"]
    if len(inventory) < 2:
        return pd.DataFrame(columns=columns)

    thumbnails = np.stack(
        [load_thumbnail(path) for path in inventory["filepath"]]
    )
    distances, neighbors = (
        NearestNeighbors(n_neighbors=2).fit(thumbnails).kneighbors(thumbnails)
    )
    # Le premier voisin est l'image elle-même, sauf si une autre miniature
    # lui est strictement identique.
    is_self_first = neighbors[:, 0] == np.arange(len(inventory))
    nearest = np.where(is_self_first, neighbors[:, 1], neighbors[:, 0])
    nearest_distance = np.where(is_self_first, distances[:, 1], distances[:, 0])

    image_ids = inventory["image_id"].to_numpy()
    neighbor_ids = image_ids[nearest]
    pairs = pd.DataFrame(
        {
            "image_id_a": np.minimum(image_ids, neighbor_ids),
            "image_id_b": np.maximum(image_ids, neighbor_ids),
            "distance": nearest_distance,
        }
    )
    return (
        pairs.loc[pairs["distance"] < threshold]
        .drop_duplicates(["image_id_a", "image_id_b"])
        .sort_values("distance")
        .reset_index(drop=True)
    )


def _removal_report(
    inventory: pd.DataFrame,
    removed_ids: list[str],
    kept_ids: list[str],
    reason: str,
    distances: float | list[float],
) -> pd.DataFrame:
    """Construit les lignes du rapport pour une étape de déduplication."""
    subsets = inventory.set_index("image_id")["subset"]
    return pd.DataFrame(
        {
            "image_id": removed_ids,
            "subset": subsets.loc[removed_ids].to_numpy(),
            "kept_image_id": kept_ids,
            "kept_subset": subsets.loc[kept_ids].to_numpy(),
            "reason": reason,
            "distance": distances,
        },
        columns=REMOVED_COLUMNS,
    )


def _exact_removals(
    inventory: pd.DataFrame, kept: pd.DataFrame
) -> pd.DataFrame:
    """Décrit les images retirées par `deduplicate_exact`."""
    kept_id_by_hash = kept.set_index("sha256")["image_id"]
    removed = inventory.loc[~inventory["image_id"].isin(kept["image_id"])]
    return _removal_report(
        inventory,
        removed["image_id"].tolist(),
        removed["sha256"].map(kept_id_by_hash).tolist(),
        reason="exact",
        distances=0.0,
    )


def _near_removals(inventory: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Décrit les images à retirer parmi les paires de quasi-doublons.

    Une image retirée n'apparaît qu'une fois, associée à la paire la plus
    proche. L'image conservée peut elle-même être retirée par une autre paire.
    """
    pairs = find_near_duplicate_pairs(inventory, threshold)
    is_labeled = inventory.set_index("image_id")["is_labeled"]

    def rank(image_id: str) -> tuple[bool, str]:
        return (not is_labeled[image_id], image_id)

    kept_ids, removed_ids = [], []
    for image_id_a, image_id_b in zip(
        pairs["image_id_a"], pairs["image_id_b"], strict=True
    ):
        kept, removed = sorted((image_id_a, image_id_b), key=rank)
        kept_ids.append(kept)
        removed_ids.append(removed)

    report = _removal_report(
        inventory,
        removed_ids,
        kept_ids,
        reason="near",
        distances=pairs["distance"].tolist(),
    )
    return report.drop_duplicates("image_id")


def _without(inventory: pd.DataFrame, image_ids: pd.Series) -> pd.DataFrame:
    """Retire de l'inventaire les images dont l'`image_id` est donné."""
    return inventory.loc[~inventory["image_id"].isin(image_ids)].reset_index(
        drop=True
    )


def deduplicate_near(
    inventory: pd.DataFrame,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Retire un des deux membres de chaque paire de quasi-doublons.

    La règle est la même que pour `deduplicate_exact` : la copie annotée est
    conservée en priorité, sinon celle dont l'`image_id` est le plus petit.

    Args:
        inventory: Table sans doublons exacts, avec `filepath`.
        threshold: Distance maximale entre miniatures pour une paire.

    Returns:
        L'inventaire sans quasi-doublons.
    """
    return _without(inventory, _near_removals(inventory, threshold)["image_id"])


class DeduplicationResult(NamedTuple):
    """Résultat de `deduplicate`.

    Attributes:
        manifest: Images conservées.
        removed: Une ligne par image retirée (`image_id`, `kept_image_id`,
            `reason` valant `excluded`, `exact` ou `near`, `distance` entre
            miniatures). Une image exclue à la main n'a ni image conservée ni
            distance. Une image conservée à une étape peut être retirée à la
            suivante : il faut alors suivre la chaîne `kept_image_id`.
    """

    manifest: pd.DataFrame
    removed: pd.DataFrame


def _manual_removals(excluded: pd.DataFrame) -> pd.DataFrame:
    """Décrit les images exclues à la main (aucune image conservée)."""
    return pd.DataFrame(
        {
            "image_id": excluded["image_id"].to_numpy(),
            "subset": excluded["subset"].to_numpy(),
            "kept_image_id": pd.NA,
            "kept_subset": pd.NA,
            "reason": "excluded",
            "distance": np.nan,
        },
        columns=REMOVED_COLUMNS,
    )


def deduplicate(
    inventory: pd.DataFrame,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    excluded: Mapping[str, str] = EXCLUDED_IMAGES,
) -> DeduplicationResult:
    """Exclut les images inexploitables, puis retire les doublons.

    Les images exclues sont retirées en premier, pour qu'elles ne soient jamais
    conservées comme copie d'une autre image.

    Args:
        inventory: Table produite par `add_file_hash`.
        near_duplicate_threshold: Seuil de distance des quasi-doublons.
        excluded: Identifiants des images à exclure, avec la raison.

    Returns:
        Le manifeste des images conservées et le rapport des suppressions.
    """
    excluded_rows = inventory.loc[inventory["image_id"].isin(excluded)]
    inventory = _without(inventory, excluded_rows["image_id"])
    kept = deduplicate_exact(inventory)
    near_removed = _near_removals(kept, near_duplicate_threshold)
    removed = (
        pd.concat(
            [
                _manual_removals(excluded_rows),
                _exact_removals(inventory, kept),
                near_removed,
            ],
            ignore_index=True,
        )
        .sort_values(["reason", "image_id"])
        .reset_index(drop=True)
    )
    return DeduplicationResult(
        _without(kept, near_removed["image_id"]), removed
    )


def write_manifest(
    manifest: pd.DataFrame,
    output_path: Path,
    raw_dir: Path = RAW_DATA_DIR,
) -> None:
    """Enregistre le manifeste en CSV, avec des chemins relatifs à `raw_dir`.

    Args:
        manifest: Table avec une colonne `filepath` de chemins absolus.
        output_path: Fichier CSV à écrire (les dossiers parents sont créés).
        raw_dir: Répertoire des données brutes servant de référence.
    """
    relative_paths = manifest["filepath"].map(
        lambda path: path.relative_to(raw_dir).as_posix()  # pyright: ignore[reportAttributeAccessIssue]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.assign(filepath=relative_paths).to_csv(output_path, index=False)


def read_manifest(
    manifest_path: Path, raw_dir: Path = RAW_DATA_DIR
) -> pd.DataFrame:
    """Relit un manifeste écrit par `write_manifest`.

    Args:
        manifest_path: Fichier CSV du manifeste.
        raw_dir: Répertoire des données brutes auquel les chemins sont relatifs.

    Returns:
        Le manifeste, avec des chemins absolus dans `filepath`.
    """
    manifest = pd.read_csv(
        manifest_path, dtype={"image_id": str, "sha256": str}
    )
    return manifest.assign(
        filepath=manifest["filepath"].map(lambda path: raw_dir / path)  # pyright: ignore[reportOperatorIssue]
    )


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "manifest.csv",
    report_path: Path = REPORTS_DIR / "removed_duplicates.csv",
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
):
    """Déduplique les images brutes, écrit le manifeste et le rapport."""
    logger.info(f"Inventaire des images de {raw_dir}...")
    inventory = add_file_hash(build_inventory(raw_dir))
    manifest, removed = deduplicate(inventory, near_duplicate_threshold)

    write_manifest(manifest, output_path, raw_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(report_path, index=False)
    logger.success(
        f"{len(manifest)} images conservées sur {len(inventory)} "
        f"({manifest['is_labeled'].sum()} annotées), "
        f"manifeste enregistré dans {output_path}, "
        f"{len(removed)} suppressions listées dans {report_path}"
    )


if __name__ == "__main__":
    app()
