import os
from pathlib import Path
import random

from dotenv import load_dotenv
from loguru import logger
import numpy as np

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Constantes partagées par plusieurs étapes du projet.
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Découpage des images annotées : le test et la validation sont tirés parmi
# elles (stratifiés), toutes les autres images forment l'entraînement. Sur les
# 97 annotées, 15 pour le test et 14 pour la validation laissent 68 images à
# l'entraînement, soit environ 70 % / 15 % / 15 %.
TEST_SIZE = 15
VALIDATION_SIZE = 14

# Analyse non supervisée : composantes de la PCA et nombre de clusters (deux
# classes dans le jeu annoté).
N_COMPONENTS = 50
N_CLUSTERS = 2

# Étiquette qui remplace la racine du projet dans les sorties affichées.
SAFE_ROOT_LABEL = "<PROJECT_ROOT>"
SAFE_HOME_LABEL = "~"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


def mask_path(value: str | Path) -> str:
    """Remplace les chemins personnels par des étiquettes neutres.

    La racine du projet devient `<PROJECT_ROOT>` et le dossier personnel `~`,
    pour qu'aucun chemin local n'apparaisse dans les sorties de notebook.

    Args:
        value: Chemin (ou texte contenant des chemins) à masquer.

    Returns:
        Le texte avec la racine du projet et le dossier personnel masqués.
    """
    masked = str(value).replace(str(PROJ_ROOT), SAFE_ROOT_LABEL)
    return masked.replace(str(Path.home()), SAFE_HOME_LABEL)


def set_seed(seed: int = SEED) -> None:
    """Fixe les graines de `random`, NumPy et PyTorch.

    PyTorch n'est importé que s'il est installé : le module reste léger pour
    les étapes qui n'en ont pas besoin.

    Args:
        seed: Graine commune.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return
    torch.manual_seed(seed)
