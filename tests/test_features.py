from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from brain_mri.features import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ManifestImages,
    extract_features,
    load_features,
    preprocess_equalized,
    preprocess_gray,
    preprocess_rgb,
    save_features,
)


def _gradient_image(brightness: float = 1.0) -> Image.Image:
    ramp = np.linspace(0, 255 * brightness, 64)
    return Image.fromarray(np.tile(ramp, (64, 1)).astype(np.uint8))


def _undo_normalization(tensor: torch.Tensor) -> torch.Tensor:
    """Retrouve les pixels (0 à 1) avant la normalisation ImageNet."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


class FakeExtractor(nn.Module):
    """Deux « couches » calculées simplement à partir de l'entrée."""

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        first_channel = batch[:, :1]
        weights = torch.arange(1, 5).view(1, 4, 1, 1)
        return {
            "layer3": first_channel.repeat(1, 4, 1, 1) * weights,
            "layer4": first_channel,
        }


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    """Cinq images au contenu différent."""
    paths = []
    for index in range(5):
        pixels = np.random.default_rng(index).integers(0, 256, (64, 64))
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(pixels.astype(np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    "preprocess", [preprocess_rgb, preprocess_gray, preprocess_equalized]
)
def test_preprocess_returns_a_model_input(preprocess):
    tensor = preprocess(_gradient_image())

    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_gray_removes_color():
    colored = Image.new("RGB", (32, 32), (200, 40, 40))

    gray = _undo_normalization(preprocess_gray(colored))
    rgb = _undo_normalization(preprocess_rgb(colored))

    # La normalisation ImageNet est propre à chaque canal : on la défait pour
    # comparer les pixels eux-mêmes.
    assert torch.allclose(gray[0], gray[1], atol=1e-5)
    assert torch.allclose(gray[1], gray[2], atol=1e-5)
    assert not torch.allclose(rgb[0], rgb[1], atol=1e-2)


def test_equalization_removes_global_brightness_differences():
    bright, dark = _gradient_image(1.0), _gradient_image(0.4)

    gap_gray = (preprocess_gray(bright) - preprocess_gray(dark)).abs().mean()
    gap_equalized = (
        (preprocess_equalized(bright) - preprocess_equalized(dark))
        .abs()
        .mean()
    )

    assert gap_equalized < 0.1 * gap_gray


def test_manifest_images_applies_the_transform(image_paths: list[Path]):
    dataset = ManifestImages(image_paths, preprocess_equalized)

    assert len(dataset) == 5
    assert dataset[0].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_extract_features_pools_each_layer_in_order(image_paths: list[Path]):
    features = extract_features(
        FakeExtractor(), image_paths, transform=preprocess_equalized, batch_size=2
    )

    channel_means = np.array(
        [
            preprocess_equalized(Image.open(path))[0].mean().item()
            for path in image_paths
        ]
    )
    assert features["layer3"].shape == (5, 4)
    assert features["layer4"].shape == (5, 1)
    assert features["layer3"].dtype == np.float32
    np.testing.assert_allclose(
        features["layer4"][:, 0], channel_means, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        features["layer3"],
        channel_means[:, None] * np.arange(1, 5),
        rtol=1e-4,
        atol=1e-5,
    )


def test_extract_features_does_not_depend_on_batch_size(
    image_paths: list[Path],
):
    small = extract_features(FakeExtractor(), image_paths, batch_size=2)
    large = extract_features(FakeExtractor(), image_paths, batch_size=5)

    for name in small:
        np.testing.assert_allclose(small[name], large[name], rtol=1e-5)


def test_save_and_load_features_round_trip(tmp_path: Path):
    image_ids = ["a", "b", "c"]
    features = {
        "layer3": np.arange(6, dtype=np.float32).reshape(3, 2),
        "layer4": np.ones((3, 4), dtype=np.float32),
    }

    paths = save_features(image_ids, features, tmp_path / "out")

    assert [path.name for path in paths] == [
        "features_resnet18_layer3.npz",
        "features_resnet18_layer4.npz",
    ]
    loaded_ids, loaded_matrix = load_features(paths[0])
    assert loaded_ids.tolist() == image_ids
    np.testing.assert_array_equal(loaded_matrix, features["layer3"])
