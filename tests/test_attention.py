import numpy as np
import pytest
import torch
from torch import nn

from brain_mri.attention import IMAGENET_CLASSES, feature_maps, grad_cam


class TinyNet(nn.Module):
    """Petit réseau avec deux « couches » nommées comme celles d'un ResNet."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer3 = nn.Conv2d(3, 4, 3, padding=1)
        self.layer4 = nn.Conv2d(4, 6, 3, stride=2, padding=1)
        self.head = nn.Linear(6, 5)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden = self.layer4(torch.relu(self.layer3(batch)))
        return self.head(hidden.mean(dim=(2, 3)))


class LeftSensitiveNet(nn.Module):
    """Réseau dont le score vient uniquement de la moitié gauche de l'image."""

    def __init__(self):
        super().__init__()
        self.layer3 = nn.Conv2d(3, 1, 1, bias=False)
        self.layer4 = nn.Identity()
        self.head = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.layer3.weight.fill_(1.0)
            self.head.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.layer4(self.layer3(batch)).mean(dim=(2, 3)))


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.randn(3, 3, 16, 16, generator=torch.Generator().manual_seed(1))


def test_imagenet_classes_has_the_thousand_classes():
    assert len(IMAGENET_CLASSES) == 1000


def test_feature_maps_returns_the_maps_of_the_layer(batch: torch.Tensor):
    model = TinyNet()

    maps = feature_maps(model, batch[0], "layer4")

    expected = model.layer4(torch.relu(model.layer3(batch[:1])))[0]
    assert maps.shape == (6, 8, 8)
    assert torch.allclose(maps, expected, atol=1e-6)


def test_feature_maps_removes_its_hook(batch: torch.Tensor):
    model = TinyNet()

    feature_maps(model, batch[0], "layer3")

    assert not model.layer3._forward_hooks


def test_feature_maps_rejects_an_unknown_layer(batch: torch.Tensor):
    with pytest.raises(ValueError, match="layer9"):
        feature_maps(TinyNet(), batch[0], "layer9")


def test_grad_cam_returns_one_map_per_image(batch: torch.Tensor):
    cams, predicted, scores = grad_cam(TinyNet(), batch, "layer4")

    assert cams.shape == (3, 16, 16)
    assert predicted.shape == scores.shape == (3,)


def test_grad_cam_maps_are_normalized_between_zero_and_one(
    batch: torch.Tensor,
):
    cams, _, _ = grad_cam(TinyNet(), batch, "layer3")

    assert cams.min() >= 0
    assert cams.max() <= 1 + 1e-6
    assert np.isclose(cams.max(axis=(1, 2)), 1, atol=1e-5).sum() >= 1


def test_grad_cam_predicts_the_class_with_the_highest_score(
    batch: torch.Tensor,
):
    model = TinyNet()

    _, predicted, scores = grad_cam(model, batch, "layer4")

    probabilities = torch.softmax(model(batch), dim=1)
    assert np.array_equal(predicted, probabilities.argmax(dim=1).numpy())
    np.testing.assert_allclose(
        scores, probabilities.max(dim=1).values.numpy(), atol=1e-6
    )


def test_grad_cam_removes_its_hook(batch: torch.Tensor):
    model = TinyNet()

    grad_cam(model, batch, "layer4")

    assert not model.layer4._forward_hooks


def test_grad_cam_rejects_an_unknown_layer(batch: torch.Tensor):
    with pytest.raises(ValueError, match="layer9"):
        grad_cam(TinyNet(), batch, "layer9")


def test_grad_cam_highlights_the_region_that_drives_the_score():
    image = torch.zeros(1, 3, 16, 16)
    image[:, :, :, :8] = 1.0

    cams, _, _ = grad_cam(LeftSensitiveNet(), image, "layer3")

    assert cams[0, :, :8].mean() > cams[0, :, 8:].mean()


def test_grad_cam_is_all_zero_when_the_layer_is_silent():
    image = torch.zeros(1, 3, 16, 16)

    cams, _, _ = grad_cam(LeftSensitiveNet(), image, "layer3")

    assert np.isfinite(cams).all()
    assert cams.max() == 0
