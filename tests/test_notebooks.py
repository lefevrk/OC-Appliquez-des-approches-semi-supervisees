from pathlib import Path
import re

import pytest

from brain_mri.config import PROJ_ROOT

NOTEBOOKS = sorted((PROJ_ROOT / "notebooks").glob("*.ipynb"))
# Dossier personnel de n'importe quel système (macOS, Linux, Windows).
HOME_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)[\w.-]+")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_outputs_hide_personal_paths(notebook: Path):
    text = notebook.read_text(encoding="utf-8")

    assert str(PROJ_ROOT) not in text
    assert HOME_PATH.search(text) is None
