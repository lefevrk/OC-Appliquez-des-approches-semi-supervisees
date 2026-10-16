import hashlib
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from brain_mri.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

IMAGE_PATTERN = "*.jpg"

# Sous-dossier de `data/raw/` et label associé (NA = image non annotée).
DATA_SOURCES: dict[str, tuple[Path, str | None]] = {
    "sans_label": (Path("sans_label"), None),
    "cancer": (Path("avec_labels") / "cancer", "cancer"),
    "normal": (Path("avec_labels") / "normal", "normal"),
}

INVENTORY_COLUMNS = [
    "image_id",
    "filepath",
    "subset",
    "label",
    "is_labeled",
    "file_size_bytes",
]


def build_inventory(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Construit la table de référence des images, une ligne par fichier.

    Args:
        raw_dir: Répertoire racine des données brutes.

    Returns:
        Un DataFrame trié par `subset` puis `image_id`. La colonne `label`
        vaut `<NA>` pour les images non annotées.

    Raises:
        FileNotFoundError: Si un des sous-dossiers attendus est absent.
    """
    records = []
    for subset, (relative_dir, label) in DATA_SOURCES.items():
        source_dir = raw_dir / relative_dir
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Répertoire introuvable : {source_dir}")
        records.extend(
            {
                "image_id": path.stem,
                "filepath": path,
                "subset": subset,
                "label": label if label is not None else pd.NA,
                "is_labeled": label is not None,
                "file_size_bytes": path.stat().st_size,
            }
            for path in source_dir.glob(IMAGE_PATTERN)
        )

    return (
        pd.DataFrame(records, columns=INVENTORY_COLUMNS)
        .sort_values(["subset", "image_id"])
        .reset_index(drop=True)
    )


def file_hash(filepath: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Args:
        filepath: Chemin du fichier à lire.

    Returns:
        L'empreinte hexadécimale du contenu binaire du fichier.
    """
    with filepath.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def add_file_hash(inventory: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `sha256`, utile pour repérer les doublons exacts.

    Args:
        inventory: Table produite par `build_inventory`.

    Returns:
        Une copie de `inventory` avec la colonne `sha256`.
    """
    return inventory.assign(sha256=inventory["filepath"].map(file_hash))


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "inventory.csv",
):
    """Inventorie les images brutes et enregistre la table avec leur hash."""
    logger.info(f"Inventaire des images de {raw_dir}...")
    inventory = add_file_hash(build_inventory(raw_dir))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_path, index=False)
    logger.success(f"{len(inventory)} images enregistrées dans {output_path}")


if __name__ == "__main__":
    app()
