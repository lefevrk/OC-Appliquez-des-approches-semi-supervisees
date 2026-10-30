from pathlib import Path
import random

import numpy as np

from brain_mri import config
from brain_mri.config import (
    PROJ_ROOT,
    SAFE_HOME_LABEL,
    SAFE_ROOT_LABEL,
    mask_path,
    set_seed,
)


def test_mask_path_hides_the_project_root():
    path = PROJ_ROOT / "data" / "raw" / "image.jpg"

    assert mask_path(path) == f"{SAFE_ROOT_LABEL}/data/raw/image.jpg"


def test_mask_path_hides_the_project_root_inside_a_text():
    message = f"Lecture de {PROJ_ROOT}/data/interim/manifest.csv terminée"

    masked = mask_path(message)

    assert str(PROJ_ROOT) not in masked
    assert SAFE_ROOT_LABEL in masked


def test_mask_path_hides_the_home_directory():
    path = Path.home() / "ailleurs" / "fichier.txt"

    assert mask_path(path) == f"{SAFE_HOME_LABEL}/ailleurs/fichier.txt"


def test_mask_path_keeps_an_unrelated_path():
    assert mask_path("/opt/autre/chemin") == "/opt/autre/chemin"


def test_set_seed_makes_random_numbers_reproducible():
    set_seed(7)
    first = (random.random(), np.random.rand())
    set_seed(7)
    second = (random.random(), np.random.rand())

    assert first == second


def test_set_seed_makes_torch_reproducible():
    import torch

    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    second = torch.rand(3)

    assert torch.equal(first, second)


def test_split_sizes_leave_annotated_images_for_training():
    # 97 images annotées après déduplication : il doit en rester pour apprendre.
    assert config.TEST_SIZE + config.VALIDATION_SIZE < 97
