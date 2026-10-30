import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from brain_mri.config import N_COMPONENTS, SEED

RANDOM_STATE = SEED


def _check_fit_mask(features: np.ndarray, fit_mask: np.ndarray) -> None:
    """Vérifie que le masque désigne au moins une ligne de `features`."""
    if fit_mask.shape != (len(features),):
        raise ValueError(
            f"Le masque doit avoir une valeur par image : {len(features)} "
            f"attendues, {fit_mask.shape} reçues."
        )
    if not fit_mask.any():
        raise ValueError("Le masque d'ajustement ne contient aucune image.")


def fit_standardizer(
    features: np.ndarray, fit_mask: np.ndarray
) -> StandardScaler:
    """Ajuste un `StandardScaler` sur les images de `fit_mask` uniquement.

    Args:
        features: Matrice (images, dimensions).
        fit_mask: Masque booléen des images qui servent à l'ajustement.

    Returns:
        Le `StandardScaler` ajusté, utilisable sur toutes les images.

    Raises:
        ValueError: Si le masque n'a pas une valeur par image, ou n'en
            sélectionne aucune.
    """
    _check_fit_mask(features, fit_mask)
    return StandardScaler().fit(features[fit_mask])


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
    return fit_standardizer(features, fit_mask).transform(features)


def fit_pca(
    features: np.ndarray,
    fit_mask: np.ndarray,
    n_components: int = N_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> PCA:
    """Ajuste une PCA sur les images de `fit_mask` uniquement.

    Args:
        features: Matrice (images, dimensions), en général déjà standardisée.
        fit_mask: Masque booléen des images qui servent à l'ajustement.
        n_components: Nombre de composantes conservées.
        random_state: Graine du solveur, pour un résultat reproductible.

    Returns:
        La PCA ajustée, utilisable sur toutes les images.

    Raises:
        ValueError: Si le masque n'a pas une valeur par image, ou n'en
            sélectionne aucune.
    """
    _check_fit_mask(features, fit_mask)
    return PCA(n_components=n_components, random_state=random_state).fit(
        features[fit_mask]
    )


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
    return fit_pca(features, fit_mask, n_components, random_state).transform(
        features
    )
