from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from brain_mri.dataset import (
    DATA_SOURCES,
    EXCLUDED_IMAGES,
    INVENTORY_COLUMNS,
    INVERSE_ORIENTATIONS,
    ORIENTATIONS,
    NearDuplicateSearch,
    REMOVED_COLUMNS,
    SPLIT_NAMES,
    THUMBNAIL_SIZE,
    add_file_hash,
    build_inventory,
    deduplicate,
    deduplicate_exact,
    deduplicate_near,
    file_hash,
    find_near_duplicate_pairs,
    nearest_neighbors,
    load_thumbnail,
    read_manifest,
    read_split,
    split_dataset,
    write_manifest,
)

# Les images de test sont unies : deux nuances proches donnent des miniatures
# à distance (RMS) ~0,004 par niveau d'écart, bien en dessous de ce seuil.
TEST_NEAR_THRESHOLD = 0.01

# Seuil des copies retournées : la compression JPEG de petites images ajoute un
# bruit léger, alors que deux images différentes sont à plus de 0,2.
COPY_THRESHOLD = 0.05


def _write_image(path: Path, color: int) -> None:
    Image.new("RGB", (8, 8), (color, color, color)).save(path)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Jeu brut minimal : une image `sans_label` copie d'une `cancer`."""
    for relative_dir, _ in DATA_SOURCES.values():
        (tmp_path / relative_dir).mkdir(parents=True)

    _write_image(tmp_path / "sans_label" / "b.jpg", color=10)
    _write_image(tmp_path / "sans_label" / "a.jpg", color=20)
    _write_image(tmp_path / "avec_labels" / "cancer" / "c.jpg", color=10)
    _write_image(tmp_path / "avec_labels" / "normal" / "d.jpg", color=30)
    (tmp_path / "sans_label" / "ignored.txt").write_text("pas une image")
    return tmp_path


def test_build_inventory_lists_every_jpeg(raw_dir: Path):
    inventory = build_inventory(raw_dir)

    assert list(inventory.columns) == INVENTORY_COLUMNS
    assert inventory["image_id"].tolist() == ["c", "d", "a", "b"]
    assert inventory["subset"].tolist() == [
        "cancer",
        "normal",
        "sans_label",
        "sans_label",
    ]
    assert (inventory["file_size_bytes"] > 0).all()


def test_build_inventory_sets_labels(raw_dir: Path):
    inventory = build_inventory(raw_dir).set_index("image_id")

    assert inventory.loc["c", "label"] == "cancer"
    assert inventory.loc["d", "label"] == "normal"
    assert pd.isna(inventory.loc["a", "label"])
    assert inventory["is_labeled"].to_dict() == {
        "c": True,
        "d": True,
        "a": False,
        "b": False,
    }


def test_build_inventory_raises_when_directory_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_inventory(tmp_path)


def test_add_file_hash_matches_identical_content(raw_dir: Path):
    hashed = add_file_hash(build_inventory(raw_dir)).set_index("image_id")

    assert hashed.loc["b", "sha256"] == hashed.loc["c", "sha256"]
    assert hashed["sha256"].nunique() == 3


def test_add_file_hash_does_not_mutate_input(raw_dir: Path):
    inventory = build_inventory(raw_dir)

    add_file_hash(inventory)

    assert "sha256" not in inventory.columns


def test_file_hash_is_stable_sha256(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")

    assert file_hash(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.fixture
def dirty_inventory(tmp_path: Path) -> pd.DataFrame:
    """Inventaire avec doublons exacts et quasi-doublons.

    - exact : `x1`, `x2` (sans_label) et `c1` (cancer) ; `n1`, `n2` (normal)
    - quasi : `p1` (sans_label) et `q1` (normal)
    - seule : `y1` (sans_label)
    """
    for relative_dir, _ in DATA_SOURCES.values():
        (tmp_path / relative_dir).mkdir(parents=True)

    colors = {
        "sans_label/x1": 10,
        "sans_label/x2": 10,
        "avec_labels/cancer/c1": 10,
        "avec_labels/normal/n1": 100,
        "avec_labels/normal/n2": 100,
        "sans_label/p1": 150,
        "avec_labels/normal/q1": 151,
        "sans_label/y1": 220,
    }
    for name, color in colors.items():
        _write_image(tmp_path / f"{name}.jpg", color)
    return add_file_hash(build_inventory(tmp_path))


def test_deduplicate_exact_prefers_labeled_copy(dirty_inventory: pd.DataFrame):
    deduplicated = deduplicate_exact(dirty_inventory)

    assert "c1" in deduplicated["image_id"].tolist()
    assert not {"x1", "x2"} & set(deduplicated["image_id"])


def test_deduplicate_exact_keeps_smallest_id_among_equals(
    dirty_inventory: pd.DataFrame,
):
    deduplicated = deduplicate_exact(dirty_inventory)

    assert "n1" in deduplicated["image_id"].tolist()
    assert "n2" not in deduplicated["image_id"].tolist()


def test_deduplicate_exact_keeps_one_copy_per_content(
    dirty_inventory: pd.DataFrame,
):
    deduplicated = deduplicate_exact(dirty_inventory)

    assert deduplicated["sha256"].is_unique
    assert set(deduplicated["sha256"]) == set(dirty_inventory["sha256"])
    assert deduplicated["subset"].is_monotonic_increasing


def test_deduplicate_exact_does_not_mutate_input(
    dirty_inventory: pd.DataFrame,
):
    before = dirty_inventory.copy()

    deduplicate_exact(dirty_inventory)

    pd.testing.assert_frame_equal(dirty_inventory, before)


def test_load_thumbnail_shape_and_range(raw_dir: Path):
    thumbnail = load_thumbnail(raw_dir / "sans_label" / "a.jpg")

    assert thumbnail.shape == (THUMBNAIL_SIZE * THUMBNAIL_SIZE,)
    assert 0 <= thumbnail.min() <= thumbnail.max() <= 1


def test_find_near_duplicate_pairs_returns_ordered_pairs(
    dirty_inventory: pd.DataFrame,
):
    pairs = find_near_duplicate_pairs(
        deduplicate_exact(dirty_inventory), TEST_NEAR_THRESHOLD
    )

    assert pairs[["image_id_a", "image_id_b"]].values.tolist() == [["p1", "q1"]]
    assert (pairs["distance"] < TEST_NEAR_THRESHOLD).all()


def test_find_near_duplicate_pairs_handles_tiny_inventory(
    dirty_inventory: pd.DataFrame,
):
    pairs = find_near_duplicate_pairs(dirty_inventory.head(1))

    assert pairs.empty
    assert list(pairs.columns) == [
        "image_id_a",
        "image_id_b",
        "distance",
        "transform",
    ]


def test_deduplicate_near_prefers_labeled_copy(dirty_inventory: pd.DataFrame):
    deduplicated = deduplicate_near(
        deduplicate_exact(dirty_inventory), TEST_NEAR_THRESHOLD
    )

    assert "q1" in deduplicated["image_id"].tolist()
    assert "p1" not in deduplicated["image_id"].tolist()


def test_deduplicate_removes_exact_then_near_duplicates(
    dirty_inventory: pd.DataFrame,
):
    manifest, _ = deduplicate(dirty_inventory, TEST_NEAR_THRESHOLD)

    assert sorted(manifest["image_id"]) == ["c1", "n1", "q1", "y1"]
    assert manifest["is_labeled"].sum() == 3


def test_deduplicate_reports_every_removed_image(
    dirty_inventory: pd.DataFrame,
):
    manifest, removed = deduplicate(dirty_inventory, TEST_NEAR_THRESHOLD)

    assert list(removed.columns) == REMOVED_COLUMNS
    assert sorted(removed["image_id"]) == ["n2", "p1", "x1", "x2"]
    assert len(manifest) + len(removed) == len(dirty_inventory)


def test_deduplicate_report_explains_each_removal(
    dirty_inventory: pd.DataFrame,
):
    _, removed = deduplicate(dirty_inventory, TEST_NEAR_THRESHOLD)
    removed = removed.set_index("image_id")

    assert removed.loc["x1", "kept_image_id"] == "c1"
    assert removed.loc["x2", "kept_image_id"] == "c1"
    assert removed.loc["n2", "kept_image_id"] == "n1"
    assert removed.loc["p1", "kept_image_id"] == "q1"
    assert removed.loc["x1", "reason"] == "exact"
    assert removed.loc["p1", "reason"] == "near"
    assert removed.loc["x1", "kept_subset"] == "cancer"
    assert removed.loc["p1", "subset"] == "sans_label"
    assert removed.loc["x1", "distance"] == 0
    assert 0 <= removed.loc["p1", "distance"] < TEST_NEAR_THRESHOLD
    assert removed.loc["x1", "transform"] == "identity"
    assert removed.loc["p1", "transform"] == "identity"


def test_deduplicate_report_is_empty_without_duplicates(raw_dir: Path):
    inventory = add_file_hash(build_inventory(raw_dir))
    distinct = deduplicate_exact(inventory).query("image_id != 'b'")

    manifest, removed = deduplicate(distinct, TEST_NEAR_THRESHOLD)

    assert removed.empty
    assert list(removed.columns) == REMOVED_COLUMNS
    assert len(manifest) == len(distinct)


def test_write_manifest_stores_relative_posix_paths(
    raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
):
    inventory = add_file_hash(build_inventory(raw_dir))
    manifest_path = tmp_path_factory.mktemp("interim") / "nested" / "m.csv"

    write_manifest(inventory, manifest_path, raw_dir)

    stored = pd.read_csv(manifest_path)
    assert stored["filepath"].tolist() == [
        "avec_labels/cancer/c.jpg",
        "avec_labels/normal/d.jpg",
        "sans_label/a.jpg",
        "sans_label/b.jpg",
    ]


def test_manifest_round_trip_preserves_data(
    raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
):
    inventory = add_file_hash(build_inventory(raw_dir))
    manifest_path = tmp_path_factory.mktemp("interim") / "m.csv"

    write_manifest(inventory, manifest_path, raw_dir)

    pd.testing.assert_frame_equal(
        read_manifest(manifest_path, raw_dir), inventory
    )


def test_read_manifest_follows_moved_raw_dir(
    raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
):
    inventory = add_file_hash(build_inventory(raw_dir))
    manifest_path = tmp_path_factory.mktemp("interim") / "m.csv"
    write_manifest(inventory, manifest_path, raw_dir)
    moved_dir = tmp_path_factory.mktemp("moved") / "raw"
    shutil.copytree(raw_dir, moved_dir)

    manifest = read_manifest(manifest_path, moved_dir)

    assert all(path.is_relative_to(moved_dir) for path in manifest["filepath"])
    assert all(path.exists() for path in manifest["filepath"])


def test_excluded_images_each_have_a_reason():
    assert all(reason.strip() for reason in EXCLUDED_IMAGES.values())


def test_deduplicate_excludes_listed_images(dirty_inventory: pd.DataFrame):
    manifest, removed = deduplicate(
        dirty_inventory, TEST_NEAR_THRESHOLD, excluded={"y1": "illisible"}
    )

    excluded_row = removed.set_index("image_id").loc["y1"]
    assert "y1" not in manifest["image_id"].tolist()
    assert excluded_row["reason"] == "excluded"
    assert pd.isna(excluded_row["kept_image_id"])
    assert pd.isna(excluded_row["distance"])
    assert len(manifest) + len(removed) == len(dirty_inventory)


def test_deduplicate_excludes_before_removing_duplicates(
    dirty_inventory: pd.DataFrame,
):
    manifest, removed = deduplicate(
        dirty_inventory, TEST_NEAR_THRESHOLD, excluded={"n1": "illisible"}
    )

    # `n1` est exclue : sa copie `n2` doit être conservée, pas retirée.
    assert "n2" in manifest["image_id"].tolist()
    assert "n2" not in removed["image_id"].tolist()
    assert removed.set_index("image_id").loc["n1", "reason"] == "excluded"


def test_deduplicate_without_listed_images_excludes_nothing(
    dirty_inventory: pd.DataFrame,
):
    _, removed = deduplicate(dirty_inventory, TEST_NEAR_THRESHOLD, excluded={})

    assert "excluded" not in set(removed["reason"])


@pytest.fixture
def labeled_manifest() -> pd.DataFrame:
    """20 `cancer`, 20 `normal` annotées et 100 images non annotées."""
    labels = ["cancer"] * 20 + ["normal"] * 20 + [None] * 100
    return pd.DataFrame(
        {
            "image_id": [f"image-{index:03d}" for index in range(len(labels))],
            "label": labels,
            "is_labeled": [label is not None for label in labels],
        }
    )


def _split_of(split: pd.DataFrame, manifest: pd.DataFrame, name: str):
    """Lignes du manifeste affectées au jeu `name`."""
    ids = split.loc[split["split"] == name, "image_id"]
    return manifest[manifest["image_id"].isin(ids)]


def test_split_dataset_reserves_the_requested_sizes(
    labeled_manifest: pd.DataFrame,
):
    split = split_dataset(labeled_manifest, test_size=6, validation_size=4)

    assert split["split"].value_counts().to_dict() == {
        "train": 130,
        "test": 6,
        "validation": 4,
    }


def test_split_dataset_covers_every_image_once(
    labeled_manifest: pd.DataFrame,
):
    split = split_dataset(labeled_manifest, test_size=6, validation_size=4)

    assert split["image_id"].tolist() == labeled_manifest["image_id"].tolist()
    assert set(split["split"]) <= set(SPLIT_NAMES)


def test_split_dataset_draws_test_and_validation_among_labeled_only(
    labeled_manifest: pd.DataFrame,
):
    split = split_dataset(labeled_manifest, test_size=6, validation_size=4)

    for name in ("test", "validation"):
        assert _split_of(split, labeled_manifest, name)["is_labeled"].all()
    train = _split_of(split, labeled_manifest, "train")
    assert (~train["is_labeled"]).sum() == 100
    assert train["is_labeled"].sum() == 30


def test_split_dataset_is_stratified(labeled_manifest: pd.DataFrame):
    split = split_dataset(labeled_manifest, test_size=6, validation_size=4)

    for name, per_class in (("test", 3), ("validation", 2)):
        counts = _split_of(split, labeled_manifest, name)[
            "label"
        ].value_counts()
        assert counts.to_dict() == {"cancer": per_class, "normal": per_class}


def test_split_dataset_is_reproducible(labeled_manifest: pd.DataFrame):
    first = split_dataset(labeled_manifest, test_size=6, validation_size=4)
    second = split_dataset(labeled_manifest, test_size=6, validation_size=4)
    other_seed = split_dataset(
        labeled_manifest, test_size=6, validation_size=4, random_state=1
    )

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other_seed)


def test_split_dataset_does_not_mutate_input(labeled_manifest: pd.DataFrame):
    before = labeled_manifest.copy()

    split_dataset(labeled_manifest, test_size=6, validation_size=4)

    pd.testing.assert_frame_equal(labeled_manifest, before)


def test_split_dataset_raises_when_labeled_images_are_too_few(
    labeled_manifest: pd.DataFrame,
):
    with pytest.raises(ValueError, match="trop peu"):
        split_dataset(labeled_manifest, test_size=30, validation_size=10)


def test_split_round_trip_preserves_data(
    labeled_manifest: pd.DataFrame, tmp_path: Path
):
    split = split_dataset(labeled_manifest, test_size=6, validation_size=4)
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)

    pd.testing.assert_frame_equal(read_split(split_path), split)


# --- Quasi-doublons retournés ou pivotés ---------------------------------------


def _pattern(seed: int = 0, side: int = 32) -> np.ndarray:
    """Image sans aucune symétrie : ses 8 orientations sont toutes différentes."""
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, 256, (side, side)).astype(np.uint8)
    pattern[:8, :16] = 255
    pattern[24:, 4:10] = 0
    return pattern


def _make_source_dirs(root: Path) -> None:
    """Crée les sous-dossiers attendus par `build_inventory`."""
    for relative_dir, _ in DATA_SOURCES.values():
        (root / relative_dir).mkdir(parents=True, exist_ok=True)


def _write_pattern(path: Path, pattern: np.ndarray) -> None:
    Image.fromarray(pattern).save(path, quality=100)


def _inventory_with_copy(
    root: Path, orientation: str, copy_first: bool = False
) -> pd.DataFrame:
    """Une image `sans_label`, sa copie retournée ou pivotée, et une autre image."""
    _make_source_dirs(root)
    base = _pattern(seed=1)
    copy_id, original_id = ("a", "b") if copy_first else ("b", "a")
    _write_pattern(root / "sans_label" / f"{original_id}.jpg", base)
    _write_pattern(
        root / "sans_label" / f"{copy_id}.jpg",
        np.ascontiguousarray(ORIENTATIONS[orientation](base)),
    )
    _write_pattern(root / "sans_label" / "z.jpg", _pattern(seed=2))
    return add_file_hash(build_inventory(root))


def test_orientations_are_the_eight_symmetries_of_the_square():
    pattern = _pattern()

    results = {
        name: np.ascontiguousarray(orient(pattern)).tobytes()
        for name, orient in ORIENTATIONS.items()
    }

    assert len(ORIENTATIONS) == 8
    assert len(set(results.values())) == 8
    assert results["identity"] == pattern.tobytes()


@pytest.mark.parametrize("name", list(ORIENTATIONS))
def test_inverse_orientation_undoes_each_orientation(name: str):
    pattern = _pattern()

    restored = ORIENTATIONS[INVERSE_ORIENTATIONS.get(name, name)](
        ORIENTATIONS[name](pattern)
    )

    assert np.array_equal(restored, pattern)


@pytest.mark.parametrize("copy_first", [False, True])
@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
def test_find_near_duplicate_pairs_detects_every_orientation(
    tmp_path: Path, orientation: str, copy_first: bool
):
    inventory = _inventory_with_copy(tmp_path, orientation, copy_first)

    pairs = find_near_duplicate_pairs(inventory, COPY_THRESHOLD)

    assert pairs[["image_id_a", "image_id_b"]].values.tolist() == [["a", "b"]]
    assert pairs.loc[0, "distance"] < COPY_THRESHOLD


@pytest.mark.parametrize("copy_first", [False, True])
@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
def test_find_near_duplicate_pairs_gives_the_orientation_of_image_a(
    tmp_path: Path, orientation: str, copy_first: bool
):
    inventory = _inventory_with_copy(tmp_path, orientation, copy_first)

    pair = find_near_duplicate_pairs(inventory, COPY_THRESHOLD).iloc[0]

    by_id = inventory.set_index("image_id")["filepath"]
    first = load_thumbnail(by_id[pair["image_id_a"]]).reshape(
        THUMBNAIL_SIZE, THUMBNAIL_SIZE
    )
    second = load_thumbnail(by_id[pair["image_id_b"]]).reshape(
        THUMBNAIL_SIZE, THUMBNAIL_SIZE
    )
    oriented = ORIENTATIONS[pair["transform"]](first)
    assert np.sqrt(np.mean((oriented - second) ** 2)) < COPY_THRESHOLD


def test_find_near_duplicate_pairs_misses_a_flip_with_identity_only(
    tmp_path: Path,
):
    inventory = _inventory_with_copy(tmp_path, "flip_h")

    pairs = find_near_duplicate_pairs(
        inventory,
        COPY_THRESHOLD,
        NearDuplicateSearch(
            orientations={"identity": ORIENTATIONS["identity"]}
        ),
    )

    assert pairs.empty


def test_find_near_duplicate_pairs_ignores_distinct_images(tmp_path: Path):
    _make_source_dirs(tmp_path)
    for seed, name in enumerate("abc"):
        _write_pattern(tmp_path / "sans_label" / f"{name}.jpg", _pattern(seed))

    pairs = find_near_duplicate_pairs(
        add_file_hash(build_inventory(tmp_path)), COPY_THRESHOLD
    )

    assert pairs.empty


def test_find_near_duplicate_pairs_never_pairs_an_image_with_itself(
    tmp_path: Path,
):
    _make_source_dirs(tmp_path)
    symmetric = np.tile(_pattern()[:, :16], (1, 2))
    symmetric = np.maximum(symmetric, symmetric[:, ::-1])
    _write_pattern(tmp_path / "sans_label" / "a.jpg", symmetric)
    _write_pattern(tmp_path / "sans_label" / "b.jpg", _pattern(seed=5))

    pairs = find_near_duplicate_pairs(
        add_file_hash(build_inventory(tmp_path)), threshold=np.inf
    )

    assert (pairs["image_id_a"] != pairs["image_id_b"]).all()


@pytest.mark.parametrize("copy_first", [False, True])
def test_deduplicate_removes_a_flipped_copy_and_reports_its_orientation(
    tmp_path: Path, copy_first: bool
):
    inventory = _inventory_with_copy(tmp_path, "rot90", copy_first)

    manifest, removed = deduplicate(inventory, COPY_THRESHOLD, excluded={})

    assert sorted(manifest["image_id"]) == ["a", "z"]
    assert removed["image_id"].tolist() == ["b"]
    row = removed.iloc[0]
    assert (row["reason"], row["kept_image_id"]) == ("near", "a")
    by_id = inventory.set_index("image_id")["filepath"]
    removed_image = load_thumbnail(by_id["b"]).reshape(
        THUMBNAIL_SIZE, THUMBNAIL_SIZE
    )
    kept_image = load_thumbnail(by_id["a"]).reshape(
        THUMBNAIL_SIZE, THUMBNAIL_SIZE
    )
    oriented = ORIENTATIONS[row["transform"]](removed_image)
    assert np.sqrt(np.mean((oriented - kept_image) ** 2)) < COPY_THRESHOLD


def test_deduplicate_report_marks_manual_exclusions_without_orientation(
    dirty_inventory: pd.DataFrame,
):
    _, removed = deduplicate(
        dirty_inventory, TEST_NEAR_THRESHOLD, excluded={"y1": "illisible"}
    )

    row = removed.set_index("image_id").loc["y1"]
    assert row["reason"] == "excluded"
    assert pd.isna(row["transform"])


# --- Réglages de la recherche ---------------------------------------------------


def test_rms_distance_does_not_depend_on_the_thumbnail_size(tmp_path: Path):
    inventory = _inventory_with_copy(tmp_path, "flip_h")

    distances = [
        find_near_duplicate_pairs(
            inventory, np.inf, NearDuplicateSearch(size)
        ).loc[0, "distance"]
        for size in (16, 32, 64)
    ]

    assert max(distances) - min(distances) < 0.02


def test_euclidean_distance_is_the_rms_times_the_thumbnail_side(tmp_path: Path):
    inventory = _inventory_with_copy(tmp_path, "rot180")
    size = 32

    rms = find_near_duplicate_pairs(
        inventory, np.inf, NearDuplicateSearch(size, rms=True)
    )
    euclidean = find_near_duplicate_pairs(
        inventory, np.inf, NearDuplicateSearch(size, rms=False)
    )

    np.testing.assert_allclose(euclidean["distance"], rms["distance"] * size)


def test_deduplicate_near_uses_the_given_search(tmp_path: Path):
    inventory = _inventory_with_copy(tmp_path, "flip_h")
    identity_only = NearDuplicateSearch(
        orientations={"identity": ORIENTATIONS["identity"]}
    )

    default = deduplicate_near(inventory, COPY_THRESHOLD)
    restricted = deduplicate_near(inventory, COPY_THRESHOLD, identity_only)

    assert len(default) == len(inventory) - 1
    assert len(restricted) == len(inventory)


def test_deduplicate_passes_the_search_to_the_near_duplicate_step(
    tmp_path: Path,
):
    inventory = _inventory_with_copy(tmp_path, "flip_v")
    identity_only = NearDuplicateSearch(
        orientations={"identity": ORIENTATIONS["identity"]}
    )

    manifest, removed = deduplicate(
        inventory, COPY_THRESHOLD, excluded={}, search=identity_only
    )

    assert len(manifest) == len(inventory)
    assert removed.empty


def test_nearest_neighbors_returns_one_row_per_image(tmp_path: Path):
    inventory = _inventory_with_copy(tmp_path, "flip_h")

    neighbors = nearest_neighbors(inventory)

    assert sorted(neighbors["image_id"]) == sorted(inventory["image_id"])
    assert (neighbors["image_id"] != neighbors["neighbor_id"]).all()
    assert list(neighbors.columns) == [
        "image_id",
        "neighbor_id",
        "distance",
        "transform",
    ]


def test_nearest_neighbors_pairs_a_copy_with_its_flipped_original(
    tmp_path: Path,
):
    inventory = _inventory_with_copy(tmp_path, "flip_h")

    neighbors = nearest_neighbors(inventory).set_index("image_id")

    assert neighbors.loc["a", "neighbor_id"] == "b"
    assert neighbors.loc["b", "neighbor_id"] == "a"
    assert neighbors.loc["a", "distance"] < COPY_THRESHOLD
    assert neighbors.loc["a", "transform"] == "flip_h"


def test_nearest_neighbors_handles_tiny_inventory(tmp_path: Path):
    inventory = _inventory_with_copy(tmp_path, "flip_h").head(1)

    assert nearest_neighbors(inventory).empty
