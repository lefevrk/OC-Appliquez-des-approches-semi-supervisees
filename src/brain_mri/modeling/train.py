from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2

from brain_mri.config import IMAGE_SIZE
from brain_mri.features import IMAGENET_MEAN, IMAGENET_STD

# Réglages de l'entraînement, fixés d'avance.
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 16
WEAK_EPOCHS = 10
MAX_EPOCHS = 30
PATIENCE = 5
DECISION_THRESHOLD = 0.5
NUM_CLASSES = 2

# Prétraitement des images : redimensionnement, puis normalisation d'ImageNet.
_RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Augmentation géométrique légère, appliquée à l'entraînement seulement.
AUGMENT = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]
)


def create_model(
    seed: int,
    device: torch.device | None = None,
    weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT,
) -> nn.Module:
    """Crée un ResNet50 dont seules `layer4` et la couche finale s'entraînent.

    Toutes les couches sont gelées, sauf `layer4` (les derniers blocs
    convolutifs) et la couche de classification, remplacée par une couche
    linéaire à deux sorties (`normal`, `cancer`).

    Args:
        seed: Graine de l'initialisation de la couche finale.
        device: Appareil de calcul (CPU par défaut).
        weights: Poids pré-entraînés (`None` pour une initialisation
            aléatoire, utile pour les tests).

    Returns:
        Le modèle, sur `device`.
    """
    torch.manual_seed(seed)
    model = resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model.to(device or torch.device("cpu"))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Compte les paramètres d'un modèle.

    Args:
        model: Modèle à décrire.

    Returns:
        Le nombre total de paramètres et le nombre de paramètres entraînables.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


class LabeledImages(Dataset):
    """Images d'une liste de fichiers, prétraitées à chaque passage."""

    def __init__(
        self,
        filepaths: Sequence[Path],
        labels: Sequence[int],
        augment: bool = False,
    ):
        """Mémorise les chemins, les classes et le mode.

        Args:
            filepaths: Chemins des images.
            labels: Classe de chaque image (1 = `cancer`, 0 = `normal`).
            augment: Applique l'augmentation à chaque passage si vrai.

        Raises:
            ValueError: S'il n'y a pas une classe par image.
        """
        if len(filepaths) != len(labels):
            raise ValueError(
                f"Une classe par image est attendue : {len(filepaths)} "
                f"images, {len(labels)} classes."
            )
        self.filepaths = list(filepaths)
        self.labels = np.asarray(labels, dtype=int)
        self.augment = augment

    def __len__(self) -> int:
        """Nombre d'images."""
        return len(self.filepaths)

    def load(self, index: int) -> torch.Tensor:
        """Charge une image en niveaux de gris, valeurs entre 0 et 1.

        Args:
            index: Position de l'image dans la liste.

        Returns:
            Un tenseur (3, 224, 224), le canal gris étant recopié sur trois
            canaux.
        """
        with Image.open(self.filepaths[index]) as image:
            return transforms.functional.to_tensor(
                _RESIZE(image.convert("L").convert("RGB"))
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Retourne l'image normalisée, augmentée si besoin, et sa classe."""
        image = self.load(index)
        if self.augment:
            image = AUGMENT(image)
        return _NORMALIZE(image), int(self.labels[index])


def make_loader(
    filepaths: Sequence[Path],
    labels: Sequence[int],
    training: bool,
    balanced: bool = False,
    seed: int = 0,
    batch_size: int = TRAIN_BATCH_SIZE,
) -> DataLoader:
    """Crée les lots d'un jeu : tirage aléatoire à l'entraînement, ordre fixe sinon.

    L'augmentation ne s'applique qu'à l'entraînement.

    Args:
        filepaths: Chemins des images.
        labels: Classe de chaque image (1 = `cancer`, 0 = `normal`).
        training: Mode entraînement (augmentation et tirage aléatoire).
        balanced: Tire les images pour que les deux classes pèsent autant
            (entraînement seulement).
        seed: Graine du tirage.
        batch_size: Nombre d'images par lot.

    Returns:
        Le chargeur de lots.

    Raises:
        ValueError: Si un tirage équilibré est demandé alors qu'une classe
            n'a aucune image.
    """
    dataset = LabeledImages(filepaths, labels, augment=training)
    if not training:
        return DataLoader(dataset, batch_size=batch_size)
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        counts = np.bincount(dataset.labels, minlength=NUM_CLASSES)
        if (counts == 0).any():
            raise ValueError(
                "Un tirage équilibré demande des images de chaque classe."
            )
        weights = (1.0 / counts)[dataset.labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prédit la probabilité de `cancer` de chaque image d'un jeu.

    Args:
        model: Modèle de classification à deux sorties.
        loader: Lots d'images et de classes.
        device: Appareil de calcul (CPU par défaut).

    Returns:
        Les classes, les probabilités de `cancer` et la perte moyenne.
    """
    device = device or torch.device("cpu")
    model.eval()
    labels, probabilities, total = [], [], 0.0
    for images, classes in loader:
        logits = model(images.to(device))
        total += F.cross_entropy(
            logits, classes.to(device), reduction="sum"
        ).item()
        probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
        labels.append(classes.numpy())
    labels = np.concatenate(labels)
    return labels, np.concatenate(probabilities), total / len(labels)


def classification_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Mesure la détection du `cancer` (classe positive).

    Args:
        labels: Classes (1 = `cancer`, 0 = `normal`).
        scores: Probabilité de `cancer` de chaque image.
        threshold: Seuil à partir duquel une image est classée `cancer`.

    Returns:
        Le rappel et la précision de `cancer`, le F1 macro (moyenne des F1
        des deux classes) et l'exactitude.
    """
    labels = labels.astype(int)
    predicted = (scores >= threshold).astype(int)
    return {
        "recall": recall_score(labels, predicted, zero_division=0),
        "precision": precision_score(labels, predicted, zero_division=0),
        "f1_macro": f1_score(
            labels, predicted, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(labels, predicted),
    }


def fit(
    model: nn.Module,
    loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    device: torch.device | None = None,
    patience: int | None = None,
    learning_rate: float = LEARNING_RATE,
) -> tuple[pd.DataFrame, int | None]:
    """Entraîne les couches libres du modèle et mesure la validation.

    Le modèle est en mode entraînement : les couches de normalisation par lot
    s'adaptent aux images. Sans `patience`, l'entraînement dure `epochs`
    époques et le modèle final est conservé. Avec `patience`, il s'arrête
    après autant d'époques sans baisse de la perte de validation, et la
    meilleure époque est restaurée.

    Args:
        model: Modèle à entraîner, modifié sur place.
        loader: Lots d'entraînement.
        validation_loader: Lots de validation, mesurés à chaque époque.
        epochs: Nombre maximal d'époques.
        device: Appareil de calcul (CPU par défaut).
        patience: Époques sans baisse avant l'arrêt (aucun arrêt anticipé si
            `None`).
        learning_rate: Taux d'apprentissage d'Adam.

    Returns:
        L'historique par époque (`epoch`, `train_loss`, `val_loss` et les
        mesures `val_*` de `classification_scores`) et le numéro de l'époque
        conservée (`None` si le modèle final est conservé).
    """
    device = device or torch.device("cpu")
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=learning_rate,
    )
    history, best, waiting = [], None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for images, classes in loader:
            images, classes = images.to(device), classes.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), classes)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(classes)
            seen += len(classes)
        labels, probabilities, validation_loss = predict(
            model, validation_loader, device
        )
        scores = classification_scores(labels, probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / seen,
                "val_loss": validation_loss,
                **{f"val_{name}": value for name, value in scores.items()},
            }
        )
        if patience is None:
            continue
        if best is None or validation_loss < best["loss"]:
            best = {
                "loss": validation_loss,
                "epoch": epoch,
                "state": deepcopy(model.state_dict()),
            }
            waiting = 0
        else:
            waiting += 1
            if waiting == patience:
                break
    if best is not None:
        model.load_state_dict(best["state"])
    return pd.DataFrame(history), None if best is None else best["epoch"]
