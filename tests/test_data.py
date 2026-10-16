from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
import pytest

from brain_mri.dataset import (
    DATA_SOURCES,
    EXCLUDED_IMAGES,
    INVENTORY_COLUMNS,
    REMOVED_COLUMNS,
    THUMBNAIL_SIZE,
    add_file_hash,
    build_inventory,
    deduplicate,
    deduplicate_exact,
    deduplicate_near,
    file_hash,
    find_near_duplicate_pairs,
    load_thumbnail,
    read_manifest,
    write_manifest,
)

# Les images de test sont unies : deux nuances proches donnent des miniatures
# à distance ~0,12 par niveau d'écart, bien en dessous de ce seuil de test.
TEST_NEAR_THRESHOLD = 1.0


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
    assert list(pairs.columns) == ["image_id_a", "image_id_b", "distance"]


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
