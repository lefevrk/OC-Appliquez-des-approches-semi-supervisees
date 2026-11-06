from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path
from typing import NamedTuple

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import typer

from brain_mri.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
)

app = typer.Typer()

IMAGE_PATTERN = "*.jpg"

# Détection des quasi-doublons : miniatures en niveaux de gris, comparées dans
# les 8 orientations. Le seuil est un écart quadratique moyen (RMS) entre
# pixels dans [0, 1], choisi par revue visuelle des paires dans
# `notebooks/02.01-kl-data-cleaning.ipynb`.
THUMBNAIL_SIZE = 64
NEAR_DUPLICATE_THRESHOLD = 0.02

# Les 8 symétries du carré, appliquées aux deux derniers axes d'un tableau
# (une image ou un lot d'images).
ORIENTATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1, axes=(-2, -1)),
    "rot180": lambda a: np.rot90(a, 2, axes=(-2, -1)),
    "rot270": lambda a: np.rot90(a, 3, axes=(-2, -1)),
    "flip_h": lambda a: a[..., ::-1],
    "flip_v": lambda a: a[..., ::-1, :],
    "transpose": lambda a: np.swapaxes(a, -1, -2),
    "anti_transpose": lambda a: np.rot90(
        np.swapaxes(a, -1, -2), 2, axes=(-2, -1)
    ),
}
# Orientation qui annule celle-ci : les six autres sont leur propre inverse.
INVERSE_ORIENTATIONS = {"rot90": "rot270", "rot270": "rot90"}


class NearDuplicateSearch(NamedTuple):
    """Réglages de la recherche de quasi-doublons.

    Attributes:
        thumbnail_size: Côté des miniatures comparées, en pixels.
        orientations: Orientations essayées pour chaque image, par nom.
        rms: `True` pour mesurer l'écart quadratique moyen (RMS) entre
            miniatures, indépendant de leur taille ; `False` pour la distance
            euclidienne.
    """

    thumbnail_size: int = THUMBNAIL_SIZE
    orientations: Mapping[str, Callable[[np.ndarray], np.ndarray]] = (
        ORIENTATIONS
    )
    rms: bool = True


DEFAULT_SEARCH = NearDuplicateSearch()

# Images retirées à la main, car inexploitables (cf. notebook 02.01). Le fichier
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

# Découpage des images annotées : les tailles du test et de la validation sont
# dans `brain_mri.config`.
SPLIT_RANDOM_STATE = SEED
SPLIT_NAMES = ("train", "validation", "test")

REMOVED_COLUMNS = [
    "image_id",
    "subset",
    "kept_image_id",
    "kept_subset",
    "reason",
    "distance",
    "transform",
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


def _nearest_by_orientation(
    thumbnails: np.ndarray,
    orientations: Mapping[str, Callable[[np.ndarray], np.ndarray]],
) -> pd.DataFrame:
    """Cherche la plus proche voisine de chaque image, orientations comprises.

    La miniature de chaque image est retournée ou pivotée, puis comparée aux
    miniatures d'origine. Une image n'est jamais sa propre voisine, même quand
    elle est presque symétrique.

    Args:
        thumbnails: Tableau (images, côté, côté) de miniatures dans [0, 1].
        orientations: Orientations à essayer, par nom.

    Returns:
        Une ligne par image : `index_a`, `index_b` (sa voisine), `distance`
        (euclidienne entre miniatures) et `transform` (orientation appliquée
        à l'image A).
    """
    count = len(thumbnails)
    index = NearestNeighbors(n_neighbors=2).fit(thumbnails.reshape(count, -1))
    best = np.full(count, np.inf)
    neighbor = np.full(count, -1)
    transform = np.full(count, "", dtype=object)
    for name, orient in orientations.items():
        distances, found = index.kneighbors(
            orient(thumbnails).reshape(count, -1)
        )
        # Le premier voisin est l'image elle-même, sauf si une autre miniature
        # lui est strictement identique.
        self_first = found[:, 0] == np.arange(count)
        distance = np.where(self_first, distances[:, 1], distances[:, 0])
        closest = np.where(self_first, found[:, 1], found[:, 0])
        better = distance < best
        best[better] = distance[better]
        neighbor[better] = closest[better]
        transform[better] = name
    return pd.DataFrame(
        {
            "index_a": np.arange(count),
            "index_b": neighbor,
            "distance": best,
            "transform": transform,
        }
    )


def nearest_neighbors(
    inventory: pd.DataFrame,
    search: NearDuplicateSearch = DEFAULT_SEARCH,
) -> pd.DataFrame:
    """Cherche la plus proche voisine de chaque image.

    Chaque image est comparée à ses voisines dans toutes les orientations de
    `search` : avec les 8 symétries du carré, une copie retournée ou pivotée
    est détectée comme une copie dans l'orientation d'origine. Les doublons
    exacts doivent être retirés au préalable.

    Args:
        inventory: Table avec les colonnes `image_id` et `filepath`.
        search: Taille des miniatures, orientations et mesure de la distance.

    Returns:
        Une ligne par image : `image_id`, `neighbor_id`, `distance` (RMS ou
        distance euclidienne, selon `search`) et `transform`, l'orientation à
        appliquer à l'image pour retrouver sa voisine. Vide s'il y a moins de
        deux images.
    """
    columns = ["image_id", "neighbor_id", "distance", "transform"]
    if len(inventory) < 2:
        return pd.DataFrame(columns=columns)

    size = search.thumbnail_size
    thumbnails = np.stack(
        [load_thumbnail(path, size) for path in inventory["filepath"]]
    ).reshape(-1, size, size)
    neighbors = _nearest_by_orientation(thumbnails, search.orientations)
    # Le RMS est la distance euclidienne divisée par la racine du nombre de
    # pixels, soit le côté de la miniature.
    distance = (
        neighbors["distance"] / size if search.rms else neighbors["distance"]
    )
    ids = inventory["image_id"].to_numpy()
    return pd.DataFrame(
        {
            "image_id": ids[neighbors["index_a"]],
            "neighbor_id": ids[neighbors["index_b"]],
            "distance": distance,
            "transform": neighbors["transform"],
        },
        columns=columns,
    )


def find_near_duplicate_pairs(
    inventory: pd.DataFrame,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    search: NearDuplicateSearch = DEFAULT_SEARCH,
) -> pd.DataFrame:
    """Repère les paires d'images quasi identiques.

    Une paire réunit une image et sa plus proche voisine (voir
    `nearest_neighbors`) ; elle n'est comptée qu'une fois, même si les deux
    images sont voisines l'une de l'autre.

    Args:
        inventory: Table avec les colonnes `image_id` et `filepath`.
        threshold: Distance en dessous de laquelle une paire est retenue, dans
            l'unité de `search` (RMS ou distance euclidienne).
        search: Taille des miniatures, orientations et mesure de la distance.

    Returns:
        Un DataFrame `image_id_a`, `image_id_b`, `distance`, `transform`, avec
        `image_id_a < image_id_b`, trié par distance croissante. `transform`
        est l'orientation à appliquer à l'image A pour retrouver l'image B.
    """
    columns = ["image_id_a", "image_id_b", "distance", "transform"]
    neighbors = nearest_neighbors(inventory, search)
    if neighbors.empty:
        return pd.DataFrame(columns=columns)

    swapped = (neighbors["image_id"] > neighbors["neighbor_id"]).to_numpy()
    pairs = pd.DataFrame(
        {
            "image_id_a": np.where(
                swapped, neighbors["neighbor_id"], neighbors["image_id"]
            ),
            "image_id_b": np.where(
                swapped, neighbors["image_id"], neighbors["neighbor_id"]
            ),
            "distance": neighbors["distance"],
            # Si les images changent de rôle, l'orientation s'inverse.
            "transform": [
                INVERSE_ORIENTATIONS.get(name, name) if flip else name
                for name, flip in zip(
                    neighbors["transform"], swapped, strict=True
                )
            ],
        },
        columns=columns,
    )
    return (
        pairs[pairs["distance"] < threshold]
        .sort_values("distance")
        .drop_duplicates(["image_id_a", "image_id_b"])
        .reset_index(drop=True)
    )


def _removal_report(
    inventory: pd.DataFrame,
    removed_ids: list[str],
    kept_ids: list[str],
    reason: str,
    distances: float | list[float],
    transforms: str | list[str] = "identity",
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
            "transform": transforms,
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


def _near_removals(
    inventory: pd.DataFrame, threshold: float, search: NearDuplicateSearch
) -> pd.DataFrame:
    """Décrit les images à retirer parmi les paires de quasi-doublons.

    Une image retirée n'apparaît qu'une fois, associée à la paire la plus
    proche. L'image conservée peut elle-même être retirée par une autre paire.
    La colonne `transform` donne l'orientation à appliquer à l'image retirée
    pour retrouver l'image conservée.
    """
    pairs = find_near_duplicate_pairs(inventory, threshold, search)
    is_labeled = inventory.set_index("image_id")["is_labeled"]

    def rank(image_id: str) -> tuple[bool, str]:
        return (not is_labeled[image_id], image_id)

    kept_ids, removed_ids, transforms = [], [], []
    for image_id_a, image_id_b, transform in zip(
        pairs["image_id_a"],
        pairs["image_id_b"],
        pairs["transform"],
        strict=True,
    ):
        kept, removed = sorted((image_id_a, image_id_b), key=rank)
        kept_ids.append(kept)
        removed_ids.append(removed)
        # `transform` s'applique à l'image A : si c'est l'image conservée,
        # l'orientation de l'image retirée est l'inverse.
        transforms.append(
            transform
            if removed == image_id_a
            else INVERSE_ORIENTATIONS.get(transform, transform)
        )

    report = _removal_report(
        inventory,
        removed_ids,
        kept_ids,
        reason="near",
        distances=pairs["distance"].tolist(),
        transforms=transforms,
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
    search: NearDuplicateSearch = DEFAULT_SEARCH,
) -> pd.DataFrame:
    """Retire un des deux membres de chaque paire de quasi-doublons.

    La règle est la même que pour `deduplicate_exact` : la copie annotée est
    conservée en priorité, sinon celle dont l'`image_id` est le plus petit.

    Args:
        inventory: Table sans doublons exacts, avec `filepath`.
        threshold: Distance maximale entre miniatures pour une paire, dans
            l'unité de `search`.
        search: Taille des miniatures, orientations et mesure de la distance.

    Returns:
        L'inventaire sans quasi-doublons.
    """
    return _without(
        inventory, _near_removals(inventory, threshold, search)["image_id"]
    )


class DeduplicationResult(NamedTuple):
    """Résultat de `deduplicate`.

    Attributes:
        manifest: Images conservées.
        removed: Une ligne par image retirée (`image_id`, `kept_image_id`,
            `reason` valant `excluded`, `exact` ou `near`, `distance` RMS entre
            miniatures, `transform` : orientation de l'image retirée pour
            retrouver l'image conservée). Une image exclue à la main n'a ni
            image conservée, ni distance, ni orientation. Une image conservée à
            une étape peut être retirée à la suivante : il faut alors suivre la
            chaîne `kept_image_id`.
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
            "transform": pd.NA,
        },
        columns=REMOVED_COLUMNS,
    )


def deduplicate(
    inventory: pd.DataFrame,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    excluded: Mapping[str, str] = EXCLUDED_IMAGES,
    search: NearDuplicateSearch = DEFAULT_SEARCH,
) -> DeduplicationResult:
    """Exclut les images inexploitables, puis retire les doublons.

    Les images exclues sont retirées en premier, pour qu'elles ne soient jamais
    conservées comme copie d'une autre image.

    Args:
        inventory: Table produite par `add_file_hash`.
        near_duplicate_threshold: Seuil de distance des quasi-doublons, dans
            l'unité de `search`.
        excluded: Identifiants des images à exclure, avec la raison.
        search: Taille des miniatures, orientations et mesure de la distance.

    Returns:
        Le manifeste des images conservées et le rapport des suppressions.
    """
    excluded_rows = inventory.loc[inventory["image_id"].isin(excluded)]
    inventory = _without(inventory, excluded_rows["image_id"])
    kept = deduplicate_exact(inventory)
    near_removed = _near_removals(kept, near_duplicate_threshold, search)
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


def split_dataset(
    manifest: pd.DataFrame,
    test_size: int = TEST_SIZE,
    validation_size: int = VALIDATION_SIZE,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:
    """Répartit les images en entraînement, validation et test.

    Le test et la validation sont tirés **parmi les images annotées
    uniquement**, car il faut des labels forts pour les évaluer. Les tirages
    sont stratifiés : les classes y sont aussi équilibrées que possible. Toutes
    les autres images, annotées ou non, forment l'entraînement. Les images non
    annotées ne forment aucune classe : elles servent à l'apprentissage non
    supervisé.

    Args:
        manifest: Table avec les colonnes `image_id`, `label` et `is_labeled`.
        test_size: Nombre d'images annotées réservées au test.
        validation_size: Nombre d'images annotées réservées à la validation.
        random_state: Graine du tirage, pour un découpage reproductible.

    Returns:
        Une ligne par image du manifeste (`image_id`, `split`), `split` valant
        `train`, `validation` ou `test`.

    Raises:
        ValueError: Si les images annotées ne suffisent pas à réserver le test
            et la validation tout en gardant des images annotées à
            l'entraînement.
    """
    annotated = manifest[manifest["is_labeled"]]
    if test_size + validation_size >= len(annotated):
        raise ValueError(
            f"{len(annotated)} images annotées : trop peu pour réserver "
            f"{test_size} images de test et {validation_size} de validation."
        )
    remaining_ids, test_ids = train_test_split(
        annotated["image_id"],
        test_size=test_size,
        stratify=annotated["label"],
        random_state=random_state,
    )
    remaining = annotated[annotated["image_id"].isin(remaining_ids)]
    _, validation_ids = train_test_split(
        remaining["image_id"],
        test_size=validation_size,
        stratify=remaining["label"],
        random_state=random_state,
    )
    split = np.select(
        [
            manifest["image_id"].isin(test_ids),
            manifest["image_id"].isin(validation_ids),
        ],
        ["test", "validation"],
        default="train",
    )
    return pd.DataFrame({"image_id": manifest["image_id"], "split": split})


def read_split(split_path: Path) -> pd.DataFrame:
    """Relit un découpage écrit par `main`.

    Args:
        split_path: Fichier CSV du découpage.

    Returns:
        Une ligne par image (`image_id`, `split`).
    """
    return pd.read_csv(split_path, dtype={"image_id": str, "split": str})


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "manifest.csv",
    report_path: Path = REPORTS_DIR / "removed_duplicates.csv",
    split_path: Path = PROCESSED_DATA_DIR / "split.csv",
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    test_size: int = TEST_SIZE,
    validation_size: int = VALIDATION_SIZE,
):
    """Déduplique les images brutes, écrit le manifeste, le rapport, le découpage."""
    logger.info(f"Inventaire des images de {raw_dir}...")
    inventory = add_file_hash(build_inventory(raw_dir))
    manifest, removed = deduplicate(inventory, near_duplicate_threshold)

    write_manifest(manifest, output_path, raw_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(report_path, index=False)

    split = split_dataset(manifest, test_size, validation_size)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False)

    logger.success(
        f"{len(manifest)} images conservées sur {len(inventory)} "
        f"({manifest['is_labeled'].sum()} annotées), "
        f"manifeste enregistré dans {output_path}, "
        f"{len(removed)} suppressions listées dans {report_path}, "
        f"découpage ({split['split'].value_counts().to_dict()}) "
        f"enregistré dans {split_path}"
    )


if __name__ == "__main__":
    app()
