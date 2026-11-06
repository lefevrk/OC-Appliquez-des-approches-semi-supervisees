from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from brain_mri.config import IMAGE_SIZE
from brain_mri.features import preprocess_gray
from brain_mri.modeling.train import (
    LabeledImages,
    classification_scores,
    count_parameters,
    create_model,
    fit,
    make_loader,
    predict,
)


class Brightness(nn.Module):
    """Deux sorties lues dans la luminosité moyenne : sombre ou clair."""

    def __init__(self, scale: float = 1.0):
        """Crée un paramètre libre, pour que `fit` ait quelque chose à régler."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Renvoie les scores (classe 0, classe 1) de chaque image."""
        level = batch.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack([-level, level], dim=1)


def _save_images(tmp_path: Path, levels: list[int]) -> list[Path]:
    """Une image unie par niveau de gris."""
    paths = []
    for index, level in enumerate(levels):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((64, 64), level, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def two_classes(tmp_path: Path) -> tuple[list[Path], list[int]]:
    """Huit images sombres (classe 0) et huit images claires (classe 1)."""
    levels = [30] * 8 + [220] * 8
    return _save_images(tmp_path, levels), [0] * 8 + [1] * 8


def test_create_model_trains_only_layer4_and_the_classification_layer():
    model = create_model(seed=0, weights=None)

    trainable = {
        name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"layer4", "fc"}
    assert model.fc.out_features == 2


def test_create_model_initialises_the_classification_layer_from_the_seed():
    first = create_model(seed=0, weights=None).fc.weight
    same = create_model(seed=0, weights=None).fc.weight
    other = create_model(seed=1, weights=None).fc.weight

    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_count_parameters_separates_trainable_from_frozen_ones():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    assert count_parameters(model) == (13, 4)


def test_labeled_images_requires_one_class_per_image(tmp_path: Path):
    paths = _save_images(tmp_path, [10, 20])

    with pytest.raises(ValueError, match="classe par image"):
        LabeledImages(paths, [0])


def test_labeled_images_matches_the_preprocessing_of_the_features(
    two_classes,
):
    paths, labels = two_classes
    dataset = LabeledImages(paths, labels)

    image, label = dataset[3]
    with Image.open(paths[3]) as original:
        expected = preprocess_gray(original)

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(image, expected, atol=1e-5)
    assert label == 0


def test_labeled_images_augments_only_when_asked(two_classes):
    paths, labels = two_classes
    pixels = np.random.default_rng(0).integers(0, 256, (64, 64))
    Image.fromarray(pixels.astype(np.uint8)).save(paths[0])
    plain = LabeledImages(paths, labels)[0][0]
    augmented = LabeledImages(paths, labels, augment=True)

    differences = []
    for seed in range(5):
        torch.manual_seed(seed)
        differences.append((augmented[0][0] - plain).abs().max().item())

    assert max(differences) > 0.1
    assert torch.equal(LabeledImages(paths, labels)[0][0], plain)


def test_make_loader_keeps_the_order_outside_training(two_classes):
    paths, labels = two_classes

    loader = make_loader(paths, labels, training=False, batch_size=5)

    seen = torch.cat([classes for _, classes in loader])
    assert seen.tolist() == labels
    assert [len(classes) for _, classes in loader] == [5, 5, 5, 1]


def test_make_loader_draws_both_classes_equally_when_balanced(tmp_path: Path):
    paths = _save_images(tmp_path, [30] * 36 + [220] * 4)
    labels = [0] * 36 + [1] * 4

    loader = make_loader(
        paths, labels, training=True, balanced=True, seed=0, batch_size=40
    )

    drawn = torch.cat([classes for _, classes in loader])
    assert len(drawn) == 40
    assert 10 < int(drawn.sum()) < 30


def test_make_loader_refuses_a_balanced_draw_with_a_missing_class(
    tmp_path: Path,
):
    paths = _save_images(tmp_path, [30, 40])

    with pytest.raises(ValueError, match="chaque classe"):
        make_loader(paths, [0, 0], training=True, balanced=True)


def test_predict_returns_labels_probabilities_and_the_mean_loss(two_classes):
    paths, labels = two_classes
    loader = make_loader(paths, labels, training=False, batch_size=6)

    got_labels, probabilities, loss = predict(Brightness(), loader)

    assert got_labels.tolist() == labels
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert (probabilities[8:] > probabilities[:8]).all()
    assert loss > 0


def test_classification_scores_on_a_known_example():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.1, 0.6])

    result = classification_scores(labels, scores)

    assert result == {
        "recall": 0.5,
        "precision": 0.5,
        "f1_macro": 0.5,
        "accuracy": 0.5,
    }


def test_classification_scores_follow_the_threshold():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.1, 0.6])

    result = classification_scores(labels, scores, threshold=0.3)

    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)


def test_classification_scores_do_not_fail_without_any_cancer_predicted():
    labels = np.array([1, 0])

    result = classification_scores(labels, np.array([0.1, 0.2]))

    assert result["recall"] == 0.0
    assert result["precision"] == 0.0


def test_fit_records_one_row_per_epoch_and_lowers_the_training_loss(
    two_classes,
):
    paths, labels = two_classes
    train = make_loader(paths, labels, training=True, seed=0, batch_size=8)
    validation = make_loader(paths, labels, training=False)

    # Le modèle part inversé : il classe les images claires comme sombres.
    history, best = fit(
        Brightness(scale=-1.0), train, validation, epochs=4, learning_rate=0.1
    )

    assert best is None
    assert history["epoch"].tolist() == [1, 2, 3, 4]
    assert {"train_loss", "val_loss", "val_recall", "val_f1_macro"} <= set(
        history.columns
    )
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]


def test_fit_stops_after_the_patience_and_keeps_the_best_epoch(two_classes):
    paths, labels = two_classes
    train = make_loader(paths, labels, training=True, seed=0, batch_size=8)
    validation = make_loader(paths, labels, training=False)

    history, best = fit(
        Brightness(),
        train,
        validation,
        epochs=30,
        patience=2,
        learning_rate=0.0,
    )

    assert best == 1
    assert len(history) == 3
