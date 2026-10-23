import numpy as np
import pytest
from sklearn.decomposition import PCA

from brain_mri.modeling.preprocessing import reduce_with_pca, standardize

N_FIT = 60


@pytest.fixture
def features() -> np.ndarray:
    """60 images d'ajustement, puis 20 images décalées (validation, test)."""
    rng = np.random.default_rng(0)
    fitted = rng.normal(loc=5.0, scale=2.0, size=(N_FIT, 10))
    held_out = rng.normal(loc=8.0, scale=2.0, size=(20, 10))
    return np.vstack([fitted, held_out])


@pytest.fixture
def fit_mask(features: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(features), dtype=bool)
    mask[:N_FIT] = True
    return mask


def test_standardize_centers_and_scales_the_fit_images(
    features: np.ndarray, fit_mask: np.ndarray
):
    scaled = standardize(features, fit_mask)

    assert scaled.shape == features.shape
    np.testing.assert_allclose(scaled[fit_mask].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(scaled[fit_mask].std(axis=0), 1, atol=1e-6)


def test_standardize_does_not_use_the_other_images(
    features: np.ndarray, fit_mask: np.ndarray
):
    scaled = standardize(features, fit_mask)

    # Les images décalées gardent leur décalage : elles n'ont pas servi à
    # calculer la moyenne.
    assert scaled[~fit_mask].mean() > 1


def test_standardize_does_not_mutate_input(
    features: np.ndarray, fit_mask: np.ndarray
):
    before = features.copy()

    standardize(features, fit_mask)

    np.testing.assert_array_equal(features, before)


@pytest.mark.parametrize("function", [standardize, reduce_with_pca])
def test_mask_without_image_raises(function, features: np.ndarray):
    with pytest.raises(ValueError, match="aucune image"):
        function(features, np.zeros(len(features), dtype=bool))


@pytest.mark.parametrize("function", [standardize, reduce_with_pca])
def test_mask_with_wrong_length_raises(function, features: np.ndarray):
    with pytest.raises(ValueError, match="une valeur par image"):
        function(features, np.ones(3, dtype=bool))


def test_reduce_with_pca_returns_the_requested_components(
    features: np.ndarray, fit_mask: np.ndarray
):
    reduced = reduce_with_pca(features, fit_mask, n_components=3)

    assert reduced.shape == (len(features), 3)


def test_reduce_with_pca_is_fitted_on_the_mask_only(
    features: np.ndarray, fit_mask: np.ndarray
):
    reduced = reduce_with_pca(features, fit_mask, n_components=3)
    expected = PCA(n_components=3, random_state=42).fit_transform(
        features[fit_mask]
    )

    np.testing.assert_allclose(reduced[fit_mask], expected, atol=1e-6)
    np.testing.assert_allclose(reduced[fit_mask].mean(axis=0), 0, atol=1e-6)


def test_reduce_with_pca_is_reproducible(
    features: np.ndarray, fit_mask: np.ndarray
):
    first = reduce_with_pca(features, fit_mask, n_components=3)
    second = reduce_with_pca(features, fit_mask, n_components=3)

    np.testing.assert_array_equal(first, second)
