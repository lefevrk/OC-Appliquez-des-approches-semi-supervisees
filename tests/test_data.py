from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from brain_mri.dataset import (
    DATA_SOURCES,
    INVENTORY_COLUMNS,
    add_file_hash,
    build_inventory,
    file_hash,
)


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
