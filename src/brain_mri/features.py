from collections.abc import Callable
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import typer

from brain_mri.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from brain_mri.dataset import read_manifest

app = typer.Typer()

MODEL_NAME = "resnet50"
EXTRACTION_LAYERS = ("layer3", "layer4")
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Statistiques d'ImageNet, reprises des transformations officielles des poids.
_IMAGENET = ResNet50_Weights.DEFAULT.transforms()
IMAGENET_MEAN = list(_IMAGENET.mean)
IMAGENET_STD = list(_IMAGENET.std)

_to_model_input = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_rgb(image: Image.Image) -> torch.Tensor:
    """Variante de comparaison : image RGB d'origine.

    Args:
        image: Image à préparer.

    Returns:
        Un tenseur normalisé de forme (3, 224, 224).
    """
    return _to_model_input(image.convert("RGB"))


def preprocess_gray(image: Image.Image) -> torch.Tensor:
    """Variante de comparaison : niveaux de gris recopiés sur trois canaux.

    Args:
        image: Image à préparer.

    Returns:
        Un tenseur normalisé de forme (3, 224, 224).
    """
    return _to_model_input(image.convert("L").convert("RGB"))


def preprocess_equalized(image: Image.Image) -> torch.Tensor:
    """Prétraitement retenu : niveaux de gris puis égalisation d'histogramme.

    La conversion en niveaux de gris supprime les artefacts de couleur et
    l'égalisation gomme les différences globales de luminosité et de contraste,
    deux indices qui permettraient de deviner le label sans regarder la
    pathologie. L'égalisation est appliquée en pleine résolution, avant le
    redimensionnement.

    Args:
        image: Image à préparer.

    Returns:
        Un tenseur normalisé de forme (3, 224, 224).
    """
    equalized = ImageOps.equalize(image.convert("L"))
    return _to_model_input(equalized.convert("RGB"))


class ManifestImages(Dataset):
    """Images d'une liste de fichiers, prétraitées à la volée."""

    def __init__(
        self,
        filepaths,
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        """Mémorise les chemins et la transformation à appliquer."""
        self.filepaths = list(filepaths)
        self.transform = transform

    def __len__(self) -> int:
        """Nombre d'images."""
        return len(self.filepaths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retourne l'image d'indice `index`, prétraitée."""
        with Image.open(self.filepaths[index]) as image:
            return self.transform(image)


def select_device() -> torch.device:
    """Choisit le meilleur calcul disponible : CUDA, MPS (Apple) ou CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_extractor(
    device: torch.device,
    layers: tuple[str, ...] = EXTRACTION_LAYERS,
) -> nn.Module:
    """Charge un ResNet50 pré-entraîné, gelé, qui renvoie les couches voulues.

    Les poids sont téléchargés au premier appel.

    Args:
        device: Appareil de calcul.
        layers: Noms des couches du ResNet50 à extraire.

    Returns:
        Un module qui renvoie un dictionnaire couche -> cartes de features.
    """
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    extractor = create_feature_extractor(
        backbone, return_nodes={name: name for name in layers}
    )
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    return extractor.to(device)


def extract_features(
    extractor: nn.Module,
    filepaths,
    device: torch.device | None = None,
    transform: Callable[[Image.Image], torch.Tensor] = preprocess_equalized,
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Extrait un vecteur par image et par couche (average pooling global).

    L'ordre des images est conservé.

    Args:
        extractor: Module renvoyant un dictionnaire couche -> cartes (N, C, H, W).
        filepaths: Chemins des images.
        device: Appareil de calcul (CPU par défaut).
        transform: Prétraitement d'une image.
        batch_size: Nombre d'images par lot.

    Returns:
        Un dictionnaire couche -> matrice (images, canaux) de type float32.
    """
    device = device or torch.device("cpu")
    extractor.eval()
    loader = DataLoader(
        ManifestImages(filepaths, transform),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for batch in loader:
            for name, feature_map in extractor(batch.to(device)).items():
                pooled = feature_map.mean(dim=(2, 3))
                chunks.setdefault(name, []).append(pooled.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in chunks.items()}


def save_features(
    image_ids,
    features: dict[str, np.ndarray],
    output_dir: Path = PROCESSED_DATA_DIR,
) -> list[Path]:
    """Enregistre un fichier `.npz` par couche, avec les `image_id`.

    Les identifiants et la matrice sont dans le même fichier : l'alignement
    avec le manifeste ne peut pas se perdre.

    Args:
        image_ids: Identifiants des images, dans l'ordre des lignes.
        features: Dictionnaire couche -> matrice (images, canaux).
        output_dir: Dossier de sortie (créé si besoin).

    Returns:
        Les chemins des fichiers écrits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(list(image_ids)).astype(str)
    paths = []
    for name, matrix in features.items():
        path = output_dir / f"features_{MODEL_NAME}_{name}.npz"
        np.savez_compressed(path, image_id=ids, features=matrix)
        paths.append(path)
    return paths


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Relit un fichier écrit par `save_features`.

    Args:
        path: Fichier `.npz` d'une couche.

    Returns:
        Les `image_id` et la matrice de features, dans le même ordre.
    """
    with np.load(path) as saved:
        return saved["image_id"], saved["features"]


@app.command()
def main(
    manifest_path: Path = INTERIM_DATA_DIR / "manifest.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    batch_size: int = BATCH_SIZE,
):
    """Extrait les features des images du manifeste et les enregistre."""
    manifest: pd.DataFrame = read_manifest(manifest_path)
    device = select_device()
    logger.info(f"{len(manifest)} images, calcul sur {device}...")

    extractor = load_extractor(device)
    start = time.perf_counter()
    features = extract_features(
        extractor, manifest["filepath"], device, batch_size=batch_size
    )
    elapsed = time.perf_counter() - start

    paths = save_features(manifest["image_id"], features, output_dir)
    logger.success(
        f"{len(paths)} fichiers écrits dans {output_dir} "
        f"({len(manifest) / elapsed:.1f} images par seconde)"
    )


if __name__ == "__main__":
    app()
