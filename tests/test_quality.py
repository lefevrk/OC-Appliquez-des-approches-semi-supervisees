from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from brain_mri.quality import (
    BACKGROUND_LEVEL,
    COLOR_GAP_THRESHOLD,
    DESCRIPTOR_COLUMNS,
    describe_image,
    describe_images,
)


def _save_gray(path: Path, pixels: np.ndarray) -> Path:
    Image.fromarray(pixels.astype(np.uint8)).save(path, format="PNG")
    return path


def _noise(seed: int = 0, side: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (side, side))


def test_describe_image_on_a_uniform_gray_image(tmp_path: Path):
    path = _save_gray(tmp_path / "gray.png", np.full((32, 32), 100))

    descriptors = describe_image(path)

    assert descriptors["brightness"] == pytest.approx(100)
    assert descriptors["contrast"] == pytest.approx(0)
    assert descriptors["background_fraction"] == 0
    assert descriptors["channel_gap"] == 0
    assert descriptors["sharpness"] == pytest.approx(0)


def test_describe_image_counts_the_background(tmp_path: Path):
    pixels = np.full((10, 10), 200)
    pixels[:4] = 0
    path = _save_gray(tmp_path / "half.png", pixels)

    assert describe_image(path)["background_fraction"] == pytest.approx(0.4)


def test_background_level_is_exclusive(tmp_path: Path):
    path = _save_gray(tmp_path / "edge.png", np.full((8, 8), BACKGROUND_LEVEL))

    assert describe_image(path)["background_fraction"] == 0


def test_describe_image_measures_the_color_gap(tmp_path: Path):
    path = tmp_path / "color.png"
    Image.new("RGB", (16, 16), (200, 120, 100)).save(path)

    assert describe_image(path)["channel_gap"] == 80


def test_a_blurred_image_is_less_sharp(tmp_path: Path):
    sharp = _noise()
    blurred = cv2.GaussianBlur(sharp.astype(np.uint8), (9, 9), 3)

    sharp_value = describe_image(_save_gray(tmp_path / "s.png", sharp))
    blurred_value = describe_image(_save_gray(tmp_path / "b.png", blurred))

    assert blurred_value["sharpness"] < sharp_value["sharpness"] / 10


def test_describe_images_keeps_the_order_and_flags_color(tmp_path: Path):
    gray = _save_gray(tmp_path / "gray.png", np.full((8, 8), 50))
    color = tmp_path / "color.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(color)

    table = describe_images([color, gray])

    assert list(table.columns) == [*DESCRIPTOR_COLUMNS, "is_color"]
    assert table["is_color"].tolist() == [True, False]
    assert table["channel_gap"].tolist()[1] <= COLOR_GAP_THRESHOLD


def test_describe_images_returns_an_empty_table_without_images():
    table = describe_images([])

    assert table.empty
    assert list(table.columns) == [*DESCRIPTOR_COLUMNS, "is_color"]
