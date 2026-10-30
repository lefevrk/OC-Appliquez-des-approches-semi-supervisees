from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Niveau de gris en dessous duquel un pixel est compté comme du fond noir.
BACKGROUND_LEVEL = 10
# Écart maximal entre canaux RGB au-delà duquel une image est dite en couleur.
COLOR_GAP_THRESHOLD = 2

DESCRIPTOR_COLUMNS = [
    "brightness",
    "contrast",
    "background_fraction",
    "channel_gap",
    "sharpness",
]


def describe_image(filepath: Path) -> dict[str, float]:
    """Calcule des descripteurs simples du contenu d'une image.

    Args:
        filepath: Chemin de l'image.

    Returns:
        Un dictionnaire avec `brightness` (luminosité moyenne, 0 à 255),
        `contrast` (écart-type des niveaux de gris), `background_fraction`
        (part des pixels de niveau inférieur à `BACKGROUND_LEVEL`),
        `channel_gap` (écart maximal entre canaux RGB, nul pour une image en
        niveaux de gris) et `sharpness` (variance du laplacien en pleine
        résolution : plus elle est grande, plus les contours sont marqués).
    """
    with Image.open(filepath) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
        gray_levels = np.asarray(image.convert("L"))
    gray = pixels.mean(axis=2)
    channel_gap = max(
        np.abs(pixels[..., 0] - pixels[..., 1]).max(),
        np.abs(pixels[..., 1] - pixels[..., 2]).max(),
    )
    return {
        "brightness": gray.mean(),
        "contrast": gray.std(),
        "background_fraction": (gray < BACKGROUND_LEVEL).mean(),
        "channel_gap": channel_gap,
        "sharpness": cv2.Laplacian(gray_levels, cv2.CV_64F).var(),
    }


def describe_images(filepaths: Iterable[Path]) -> pd.DataFrame:
    """Calcule les descripteurs de plusieurs images, dans l'ordre donné.

    Args:
        filepaths: Chemins des images.

    Returns:
        Une ligne par image, avec les colonnes de `DESCRIPTOR_COLUMNS` et
        `is_color` (vrai si `channel_gap` dépasse `COLOR_GAP_THRESHOLD`).
    """
    descriptors = pd.DataFrame(
        [describe_image(path) for path in filepaths],
        columns=DESCRIPTOR_COLUMNS,
    )
    return descriptors.assign(
        is_color=descriptors["channel_gap"] > COLOR_GAP_THRESHOLD
    )
