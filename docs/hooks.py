"""Hooks MkDocs : publie `reports/figures/` sous `assets/figures/` dans le site."""

from pathlib import Path

from mkdocs.structure.files import File

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def on_files(files, config):
    """Ajoute les figures du dépôt aux fichiers du site."""
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        files.append(
            File.generated(
                config,
                f"assets/figures/{figure.name}",
                abs_src_path=str(figure),
            )
        )
    return files
