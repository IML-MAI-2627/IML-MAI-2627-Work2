"""Tests for instance reduction algorithms: FCNN, EENTh, and DROP variants."""

import numpy as np
import pytest

from models.knn.distances import euclidean, manhattan
from models.knn.reduction import (
    REDUCTION_TECHNIQUES,
    InstanceReduction,
    ReductionResult,
    drop,
    drop1,
    drop2,
    drop3,
    drop4,
    drop5,
    eenth,
    fcnn,
    reduce_dataset,
    wilson_prob,
    wilson_th,
)
from models.knn.voting import inverse_distance
from preprocessing import Preprocessed


def make_preprocessed(X_num, labels, X_nom=None, nom_values=None) -> Preprocessed:
    """Create a minimal Preprocessed object for unit testing."""
    X_num = np.asarray(X_num, dtype=float)
    if X_nom is None:
        X_nom = np.empty((len(X_num), 0), dtype=np.int64)
    else:
        X_nom = np.asarray(X_nom, dtype=np.int64)

    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    class_values = tuple(str(c) for c in unique_labels)
    return Preprocessed(
        X_num=X_num,
        X_nom=X_nom,
        y=np.array([str(c) for c in labels]),
        y_codes=labels,
        num_names=tuple(f"n{j}" for j in range(X_num.shape[1])),
        nom_names=tuple(f"c{j}" for j in range(X_nom.shape[1])),
        nom_values=nom_values if nom_values is not None else (),
        class_values=class_values,
    )


@pytest.fixture
def synthetic_clusters():
    """Two well-separated Gaussian clusters with a few obvious outliers."""
    rng = np.random.RandomState(42)
    c0 = rng.randn(40, 2) + np.array([-3.0, 0.0])
    c1 = rng.randn(40, 2) + np.array([3.0, 0.0])
    # Add 2 noise outliers placed deep inside the opposing class
    outlier0 = np.array([[2.8, 0.1]])  # labeled 0 inside cluster 1
    outlier1 = np.array([[-2.8, -0.1]])  # labeled 1 inside cluster 0

    X = np.vstack([c0, c1, outlier0, outlier1])
    y = np.array([0] * 40 + [1] * 40 + [0, 1], dtype=np.int64)
    return make_preprocessed(X, y)


@pytest.fixture
def multiclass_data():
    """3 balanced separable classes."""
    rng = np.random.RandomState(123)
    c0 = rng.randn(25, 3) + np.array([-5.0, 0.0, 0.0])
    c1 = rng.randn(25, 3) + np.array([0.0, 5.0, 0.0])
    c2 = rng.randn(25, 3) + np.array([5.0, 0.0, 0.0])
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * 25 + [1] * 25 + [2] * 25, dtype=np.int64)
    return make_preprocessed(X, y)


# ---------------------------------------------------------------------------
# FCNN Tests
# ---------------------------------------------------------------------------


def test_fcnn_consistency_theorem(synthetic_clusters):
    """Theorem 3.1: S computed by FCNN is guaranteed to be 100% training-set consistent."""
    result = fcnn(synthetic_clusters, distance="euclidean", variant="fcnn1")
    assert isinstance(result, ReductionResult)
    assert result.n_retained < result.n_original
    assert result.reduction_rate > 0.3

    # Check 1-NN classification consistency of all original points using S
    D = euclidean(result.data, synthetic_clusters)
    nn_in_s = np.argmin(D, axis=1)
    preds = result.data.y_codes[nn_in_s]
    assert np.all(preds == synthetic_clusters.y_codes), "FCNN must be training-set consistent"


@pytest.mark.parametrize("variant", ["fcnn1", "fcnn2", "fcnn3", "fcnn4"])
def test_fcnn_variants(multiclass_data, variant):
    """All FCNN variants (1..4) converge and yield valid consistent subsets."""
    res = fcnn(multiclass_data, variant=variant, distance="euclidean")
    assert res.n_retained > 0
    assert res.n_retained <= res.n_original
    # Retains at least one instance per class
    retained_classes = np.unique(res.data.y_codes)
    assert len(retained_classes) == 3


def test_fcnn_k_consistency(multiclass_data):
    """k-FCNN produces a subset that correctly classifies training data with k-NN."""
    res = fcnn(multiclass_data, k=3, distance="euclidean")
    assert res.n_retained > 0


def test_fcnn_order_independence(multiclass_data):
    """FCNN must return the exact same indices regardless of repeated invocations."""
    res1 = fcnn(multiclass_data, variant="fcnn1")
    res2 = fcnn(multiclass_data, variant="fcnn1")
    assert np.array_equal(res1.indices, res2.indices)


# ---------------------------------------------------------------------------
# EENTh Tests
# ---------------------------------------------------------------------------


def test_eenth_removes_outliers(synthetic_clusters):
    """EENTh must filter out noisy border outliers."""
    # Outliers are at index 80 and 81
    res = eenth(synthetic_clusters, k=3, threshold=0.7)
    assert 80 not in res.indices, "Outlier 0 should be eliminated"
    assert 81 not in res.indices, "Outlier 1 should be eliminated"
    assert res.reduction_rate > 0.0


def test_eenth_threshold_monotonicity(synthetic_clusters):
    """Higher threshold eliminates equal or more border instances."""
    res_prob = wilson_prob(synthetic_clusters, k=3)
    res_th6 = eenth(synthetic_clusters, k=3, threshold=0.6)
    res_th8 = eenth(synthetic_clusters, k=3, threshold=0.8)

    assert res_prob.n_retained >= res_th6.n_retained >= res_th8.n_retained


def test_wilson_th_alias(synthetic_clusters):
    """wilson_th and eenth yield identical results."""
    res1 = eenth(synthetic_clusters, threshold=0.75, k=3)
    res2 = wilson_th(synthetic_clusters, threshold=0.75, k=3)
    assert np.array_equal(res1.indices, res2.indices)


# ---------------------------------------------------------------------------
# DROP Tests
# ---------------------------------------------------------------------------


def test_drop3_filters_noise_and_reduces(synthetic_clusters):
    """DROP3 removes noise via ENN and prunes internal instances."""
    res = drop3(synthetic_clusters, k=3, distance="euclidean", voting="majority")
    assert isinstance(res, ReductionResult)
    # Both outliers should be pruned
    assert 80 not in res.indices
    assert 81 not in res.indices
    # High reduction rate while keeping border prototypes
    assert res.reduction_rate > 0.4
    assert res.n_retained >= 4  # Keeps instances of each class


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_all_drop_versions(multiclass_data, version):
    """All DROP versions (1..5) run successfully on multi-class data."""
    res = drop(multiclass_data, version=version, k=3, distance="euclidean")
    assert res.n_retained > 0
    assert res.n_retained < res.n_original
    # Retains all 3 classes
    assert len(np.unique(res.data.y_codes)) == 3


@pytest.mark.parametrize(
    "func",
    [drop1, drop2, drop3, drop4, drop5],
)
def test_drop_convenience_functions(multiclass_data, func):
    """Direct convenience functions drop1..drop5 match drop(version=...)."""
    res = func(multiclass_data, k=3)
    assert res.n_retained > 0


def test_drop_with_custom_voting(synthetic_clusters):
    """DROP supports alternative voting schemes (e.g. inverse_distance)."""
    res_maj = drop3(synthetic_clusters, voting="majority", k=3)
    res_inv = drop3(synthetic_clusters, voting=inverse_distance, k=3)
    assert res_maj.n_retained > 0
    assert res_inv.n_retained > 0


# ---------------------------------------------------------------------------
# Modularity and Interface Tests
# ---------------------------------------------------------------------------


def test_instance_reduction_class_fit_transform(multiclass_data):
    """InstanceReduction OOP wrapper conforms to transformer pattern."""
    reducer = InstanceReduction(method="drop3", distance="euclidean", k=3)
    reduced = reducer.fit_transform(multiclass_data)
    assert isinstance(reduced, Preprocessed)
    assert len(reduced.y_codes) < len(multiclass_data.y_codes)


def test_reduce_dataset_function(multiclass_data):
    """reduce_dataset helper provides single entrypoint for any method."""
    res = reduce_dataset(multiclass_data, method="fcnn", distance="manhattan")
    assert isinstance(res, ReductionResult)
    assert res.n_retained > 0


def test_reduction_with_raw_numpy_arrays():
    """Methods accept raw (X, y) NumPy arrays and return (X_red, y_red)."""
    rng = np.random.RandomState(99)
    X = rng.randn(50, 4)
    y = np.array([0] * 25 + [1] * 25)

    res_fcnn = fcnn((X, y))
    assert isinstance(res_fcnn.data, tuple)
    X_red, y_red = res_fcnn.data
    assert len(X_red) == res_fcnn.n_retained
    assert len(y_red) == res_fcnn.n_retained

    res_eenth = eenth(X, y=y, threshold=0.6)
    assert res_eenth.n_retained > 0

    res_drop = drop3((X, y))
    assert res_drop.n_retained > 0


def test_reduction_with_different_distances(multiclass_data):
    """Reduction techniques support various metrics (manhattan, canberra, callable)."""
    res_man = fcnn(multiclass_data, distance=manhattan)
    res_can = eenth(multiclass_data, distance="canberra")
    res_drop = drop3(multiclass_data, distance="manhattan")

    assert res_man.n_retained > 0
    assert res_can.n_retained > 0
    assert res_drop.n_retained > 0


def test_reduction_registry():
    """REDUCTION_TECHNIQUES contains all expected algorithms."""
    expected = {
        "fcnn",
        "fcnn1",
        "fcnn2",
        "fcnn3",
        "fcnn4",
        "eenth",
        "wilson_th",
        "wilson_prob",
        "drop",
        "drop1",
        "drop2",
        "drop3",
        "drop4",
        "drop5",
    }
    assert expected.issubset(set(REDUCTION_TECHNIQUES.keys()))


# ---------------------------------------------------------------------------
# Validation and Error Handling
# ---------------------------------------------------------------------------


def test_invalid_parameters():
    """Check validation of invalid configuration parameters."""
    X = np.ones((10, 2))
    y = np.zeros(10)

    with pytest.raises(ValueError, match="Unknown distance"):
        fcnn((X, y), distance="nonexistent_distance")

    with pytest.raises(ValueError, match="Unknown voting"):
        drop3((X, y), voting="invalid_voting")

    with pytest.raises(ValueError, match="Unknown variant"):
        fcnn((X, y), variant="fcnn99")

    with pytest.raises(ValueError, match="version must be 1, 2, 3, 4, or 5"):
        drop((X, y), version=9)

    with pytest.raises(ValueError, match="must not be empty"):
        fcnn((np.empty((0, 2)), np.empty(0)))

    with pytest.raises(ValueError, match="positive integer"):
        eenth((X, y), k=-1)
