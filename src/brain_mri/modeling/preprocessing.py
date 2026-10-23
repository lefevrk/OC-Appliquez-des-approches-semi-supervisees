import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_COMPONENTS = 50


def _check_fit_mask(features: np.ndarray, fit_mask: np.ndarray) -> None:
    """Vérifie que le masque désigne au moins une ligne de `features`."""
    if fit_mask.shape != (len(features),):
        raise ValueError(
            f"Le masque doit avoir une valeur par image : {len(features)} "
            f"attendues, {fit_mask.shape} reçues."
        )
    if not fit_mask.any():
        raise ValueError("Le masque d'ajustement ne contient aucune image.")


def standardize(features: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Centre et réduit chaque dimension, avec les images de `fit_mask`.

    La moyenne et l'écart-type sont calculés sur les images du masque
    uniquement, puis appliqués à **toutes** les images : celles qui n'y sont
    pas (validation, test) ne participent donc pas à l'ajustement.

    Args:
        features: Matrice (images, dimensions).
        fit_mask: Masque booléen des images qui servent à l'ajustement.

    Returns:
        La matrice standardisée, de même forme que `features`.

    Raises:
        ValueError: Si le masque n'a pas une valeur par image, ou n'en
            sélectionne aucune.
    """
    _check_fit_mask(features, fit_mask)
    scaler = StandardScaler().fit(features[fit_mask])
    return scaler.transform(features)


def reduce_with_pca(
    features: np.ndarray,
    fit_mask: np.ndarray,
    n_components: int = N_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Projette les images sur les premières composantes d'une PCA.

    La PCA est ajustée sur les images de `fit_mask` uniquement, puis appliquée
    à **toutes** les images.

    Args:
        features: Matrice (images, dimensions), en général déjà standardisée.
        fit_mask: Masque booléen des images qui servent à l'ajustement.
        n_components: Nombre de composantes conservées.
        random_state: Graine du solveur, pour un résultat reproductible.

    Returns:
        La matrice (images, `n_components`).

    Raises:
        ValueError: Si le masque n'a pas une valeur par image, ou n'en
            sélectionne aucune.
    """
    _check_fit_mask(features, fit_mask)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(features[fit_mask])
    return pca.transform(features)
