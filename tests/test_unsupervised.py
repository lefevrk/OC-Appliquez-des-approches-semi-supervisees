import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from brain_mri.modeling.unsupervised import (
    CLASS_LABELS,
    ari_without_noise,
    build_weak_labels,
    encode_labels,
    evaluate_weak_labels,
    kmeans_confidence,
    majority_mapping,
    to_weak_labels,
    wilson_interval,
)


def test_encode_labels_uses_one_for_cancer_and_nan_when_unlabeled():
    labels = pd.Series(["cancer", "normal", None])

    encoded = encode_labels(labels)

    np.testing.assert_array_equal(encoded, [1.0, 0.0, np.nan])


def test_ari_is_one_for_clusters_matching_the_labels():
    clusters = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, 0, 0])  # numéros inversés : sans effet

    assert ari_without_noise(clusters, labels) == pytest.approx(1.0)


def test_ari_ignores_noise_images():
    clusters = np.array([0, 0, 1, 1, -1, -1])
    labels = np.array([1, 1, 0, 0, 0, 1])

    assert ari_without_noise(clusters, labels) == pytest.approx(1.0)


def test_ari_is_nan_when_less_than_two_images_remain():
    clusters = np.array([0, -1, -1])
    labels = np.array([1, 0, 1])

    assert np.isnan(ari_without_noise(clusters, labels))


def test_ari_is_nan_when_remaining_images_share_one_class():
    clusters = np.array([0, 0, 1, -1])
    labels = np.array([1, 1, 1, 0])

    assert np.isnan(ari_without_noise(clusters, labels))


def test_majority_mapping_picks_the_majority_class_of_each_cluster():
    clusters = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([1, 1, 0, 0, 0, 1], dtype=float)

    mapping = majority_mapping(clusters, labels, np.ones(6, dtype=bool))

    assert mapping == {0: 1, 1: 0}


def test_majority_mapping_gives_normal_in_case_of_a_tie():
    clusters = np.array([0, 0])
    labels = np.array([1, 0], dtype=float)

    mapping = majority_mapping(clusters, labels, np.ones(2, dtype=bool))

    assert mapping == {0: 0}


def test_majority_mapping_skips_clusters_without_labeled_image():
    clusters = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, np.nan, np.nan])
    mask = np.array([True, True, False, False])

    assert majority_mapping(clusters, labels, mask) == {0: 1}


def test_majority_mapping_never_maps_the_noise():
    clusters = np.array([0, 0, -1, -1])
    labels = np.array([1, 1, 0, 0], dtype=float)

    mapping = majority_mapping(clusters, labels, np.ones(4, dtype=bool))

    assert mapping == {0: 1}


def test_to_weak_labels_is_nan_without_class():
    clusters = np.array([0, 1, 0, -1])

    weak = to_weak_labels(clusters, {0: 1})

    np.testing.assert_array_equal(weak, [1.0, np.nan, 1.0, np.nan])


@pytest.fixture
def blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.normal(loc=-5, scale=0.5, size=(30, 2)),
            rng.normal(loc=5, scale=0.5, size=(30, 2)),
        ]
    )


def test_kmeans_confidence_is_between_zero_and_one(blobs: np.ndarray):
    model = KMeans(n_clusters=2, n_init=1, random_state=0).fit(blobs)

    confidence = kmeans_confidence(model, blobs)

    assert confidence.shape == (len(blobs),)
    assert ((confidence >= 0) & (confidence <= 1)).all()


def test_kmeans_confidence_is_null_between_the_centers_and_one_on_a_center(
    blobs: np.ndarray,
):
    model = KMeans(n_clusters=2, n_init=1, random_state=0).fit(blobs)
    centers = model.cluster_centers_

    middle = kmeans_confidence(model, centers.mean(axis=0, keepdims=True))
    on_center = kmeans_confidence(model, centers[:1])

    assert middle[0] == pytest.approx(0, abs=1e-6)
    assert on_center[0] == pytest.approx(1, abs=1e-6)


def test_build_weak_labels_keeps_unlabeled_images_only():
    images = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "is_labeled": [True, False, False, False],
        }
    )

    weak = build_weak_labels(
        images,
        clusters=np.array([0, 0, 1, 2]),
        mapping={0: 1, 1: 0},
        confidence=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert weak["image_id"].tolist() == ["b", "c", "d"]
    assert list(weak.columns) == [
        "image_id",
        "cluster",
        "weak_label",
        "confidence",
        "method",
    ]
    assert weak["weak_label"].tolist()[:2] == ["cancer", "normal"]
    assert pd.isna(weak["weak_label"].iloc[2])
    assert (weak["method"] == "K-Means").all()


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(24, 30)

    assert low == pytest.approx(0.627, abs=1e-3)
    assert high == pytest.approx(0.905, abs=1e-3)


def test_wilson_interval_stays_between_zero_and_one():
    assert wilson_interval(0, 10)[0] == pytest.approx(0, abs=1e-12)
    assert wilson_interval(10, 10)[1] == pytest.approx(1, abs=1e-12)


def test_wilson_interval_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="strictement positif"):
        wilson_interval(0, 0)


def test_evaluate_weak_labels_computes_the_confusion_and_class_metrics():
    expected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    predicted = np.array([1, 1, 0, 0, 0, 0, 0, 0])

    evaluation = evaluate_weak_labels(expected, predicted)

    assert evaluation.confusion.to_numpy().tolist() == [[2, 2], [0, 4]]
    assert list(evaluation.confusion.index) == list(CLASS_LABELS)
    cancer = evaluation.per_class.loc["cancer"]
    normal = evaluation.per_class.loc["normal"]
    assert cancer["précision"] == pytest.approx(1.0)
    assert cancer["rappel"] == pytest.approx(0.5)
    assert cancer["F1"] == pytest.approx(2 / 3)
    assert normal["précision"] == pytest.approx(4 / 6)
    assert normal["rappel"] == pytest.approx(1.0)
    assert evaluation.per_class["images"].tolist() == [4, 4]


def test_evaluate_weak_labels_summarizes_accuracy_with_its_interval():
    expected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    predicted = np.array([1, 1, 0, 0, 0, 0, 0, 0])

    summary = evaluate_weak_labels(expected, predicted).summary

    assert summary["images"] == 8
    assert summary["exactitude"] == pytest.approx(0.75)
    assert summary["exactitude équilibrée"] == pytest.approx(0.75)
    assert summary["IC 95 % bas"] < 0.75 < summary["IC 95 % haut"]
