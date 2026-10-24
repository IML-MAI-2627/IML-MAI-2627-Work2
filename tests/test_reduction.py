"""Tests for instance reduction algorithms: FCNN, ENNTH, and DROP."""

import dataclasses

import numpy as np
import pytest

from models.knn.distances import euclidean, manhattan
from models.knn.knn import KNN
from models.knn.reduction import (
    REDUCTION_TECHNIQUES,
    DELReduction,
    DROPReduction,
    ENNTHReduction,
    FCNNReduction,
    InstanceReduction,
    NoReduction,
    ReductionResult,
    del_reduction,
    drop,
    ennth,
    fcnn,
    reduce_dataset,
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
    result = fcnn(synthetic_clusters, distance="euclidean", version=1)
    assert isinstance(result, ReductionResult)
    assert result.n_retained < result.n_original
    assert result.reduction_rate > 0.3

    # Check 1-NN classification consistency of all original points using S
    D = euclidean(result.data, synthetic_clusters)
    nn_in_s = np.argmin(D, axis=1)
    preds = result.data.y_codes[nn_in_s]
    assert np.all(preds == synthetic_clusters.y_codes), "FCNN must be training-set consistent"


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_fcnn_versions(multiclass_data, version):
    """All FCNN versions (1..4) converge and yield valid consistent subsets."""
    res = fcnn(multiclass_data, version=version, distance="euclidean")
    assert res.n_retained > 0
    assert res.n_retained <= res.n_original
    retained_classes = np.unique(res.data.y_codes)
    assert len(retained_classes) == 3


def test_fcnn_k_consistency(multiclass_data):
    """k-FCNN produces a subset that correctly classifies training data with k-NN."""
    res = fcnn(multiclass_data, k=3, distance="euclidean")
    assert res.n_retained > 0


def test_fcnn_order_independence(multiclass_data):
    """FCNN must return the exact same indices regardless of repeated invocations."""
    res1 = fcnn(multiclass_data, version=1)
    res2 = fcnn(multiclass_data, version=1)
    assert np.array_equal(res1.indices, res2.indices)


# ---------------------------------------------------------------------------
# ENNTH Tests
# ---------------------------------------------------------------------------


def test_ennth_removes_outliers(synthetic_clusters):
    """ENNTH must filter out noisy border outliers."""
    # Outliers are at index 80 and 81
    res = ennth(synthetic_clusters, k=3, threshold=0.7)
    assert 80 not in res.indices, "Outlier 0 should be eliminated"
    assert 81 not in res.indices, "Outlier 1 should be eliminated"
    assert res.reduction_rate > 0.0


def test_ennth_threshold_monotonicity(synthetic_clusters):
    """Higher threshold eliminates equal or more border instances."""
    res_prob = ennth(synthetic_clusters, k=3, threshold=0.0)
    res_th6 = ennth(synthetic_clusters, k=3, threshold=0.6)
    res_th8 = ennth(synthetic_clusters, k=3, threshold=0.8)

    assert res_prob.n_retained >= res_th6.n_retained >= res_th8.n_retained


# ---------------------------------------------------------------------------
# DROP Tests
# ---------------------------------------------------------------------------


def test_drop3_filters_noise_and_reduces(synthetic_clusters):
    """DROP version 3 removes noise via ENN and prunes internal instances."""
    res = drop(synthetic_clusters, version=3, k=3, distance="euclidean", voting="majority")
    assert isinstance(res, ReductionResult)
    # Both outliers should be pruned
    assert 80 not in res.indices
    assert 81 not in res.indices
    # High reduction rate while keeping border prototypes
    assert res.reduction_rate > 0.4
    assert res.n_retained >= 4


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_all_drop_versions(multiclass_data, version):
    """All DROP versions (1..5) run successfully on multi-class data."""
    res = drop(multiclass_data, version=version, k=3, distance="euclidean")
    assert res.n_retained > 0
    assert res.n_retained < res.n_original
    assert len(np.unique(res.data.y_codes)) == 3


def test_drop_with_custom_voting(synthetic_clusters):
    """DROP supports alternative voting schemes (e.g. inverse_distance)."""
    res_maj = drop(synthetic_clusters, version=3, voting="majority", k=3)
    res_inv = drop(synthetic_clusters, version=3, voting=inverse_distance, k=3)
    assert res_maj.n_retained > 0
    assert res_inv.n_retained > 0


# ---------------------------------------------------------------------------
# Modularity and Interface Tests
# ---------------------------------------------------------------------------


def test_instance_reduction_class_fit_transform(multiclass_data):
    """InstanceReduction OOP wrapper conforms to transformer pattern."""
    reducer = InstanceReduction("drop", version=3, distance="euclidean", k=3)
    reduced = reducer.fit_transform(multiclass_data)
    assert isinstance(reduced, Preprocessed)
    assert len(reduced.y_codes) < len(multiclass_data.y_codes)


def test_instance_reduction_param_validation():
    """InstanceReduction validates and accepts reduction-specific parameters."""
    # FCNN accepts version
    params_fcnn = {"reduction": "fcnn", "version": 2}
    ir_fcnn = InstanceReduction(params_fcnn)
    assert ir_fcnn.method == "fcnn"
    assert ir_fcnn.version == 2
    assert params_fcnn["version"] == 2

    # DROP accepts version
    params_drop = {"reduction": "drop", "version": 4}
    ir_drop = InstanceReduction(params_drop)
    assert ir_drop.method == "drop"
    assert ir_drop.version == 4
    assert params_drop["version"] == 4

    # ENNTH accepts threshold
    params_ennth = {"reduction": "ennth", "threshold": 0.7}
    ir_ennth = InstanceReduction(params_ennth)
    assert ir_ennth.method == "ennth"
    assert ir_ennth.threshold == 0.7
    assert params_ennth["threshold"] == 0.7

    # DEL accepts neither version nor threshold
    params_del = {"reduction": "del"}
    ir_del = InstanceReduction(params_del)
    assert ir_del.method == "del"
    assert "version" not in params_del
    assert "threshold" not in params_del

    # None accepts neither
    params_none = {"reduction": "none"}
    ir_none = InstanceReduction(params_none)
    assert ir_none.method == "none"


def test_instance_reduction_rejects_incompatible_params():
    """InstanceReduction raises ValueError on parameters incompatible with the chosen method."""
    with pytest.raises(ValueError, match="Unknown parameter 'threshold' for FCNN"):
        InstanceReduction({"reduction": "fcnn", "version": 2, "threshold": 0.5})

    with pytest.raises(ValueError, match="Unknown parameter 'threshold' for DROP"):
        InstanceReduction({"reduction": "drop", "version": 4, "threshold": 0.5})

    with pytest.raises(ValueError, match="Unknown parameter 'version' for ENNTH"):
        InstanceReduction({"reduction": "ennth", "threshold": 0.7, "version": 2})

    with pytest.raises(ValueError, match="Parameters.*are not allowed for DEL"):
        InstanceReduction({"reduction": "del", "version": 2})

    with pytest.raises(ValueError, match="Parameters.*are not allowed for DEL"):
        InstanceReduction({"reduction": "del", "threshold": 0.5})

    err_none = "are not allowed when reduction is 'none'"
    with pytest.raises(ValueError, match=err_none):
        InstanceReduction({"reduction": "none", "version": 3})

    with pytest.raises(ValueError, match=err_none):
        InstanceReduction({"reduction": "none", "threshold": 0.8})

    with pytest.raises(ValueError, match=err_none):
        InstanceReduction({"reduction": "none", "reduction_k": 3})


def test_parse_reduction_params_without_reduction():
    """Passing reduction parameters without --reduction raises ValueError."""
    from models.knn.reduction import parse_reduction_params

    msg = "without specifying the reduction type parameter '--reduction'"
    with pytest.raises(ValueError, match=msg):
        parse_reduction_params({"reduction_k": 3}, is_knn=False)

    with pytest.raises(ValueError, match=msg):
        parse_reduction_params({"version": 3}, is_knn=False)

    with pytest.raises(ValueError, match=msg):
        parse_reduction_params({"threshold": 0.5}, is_knn=False)

    with pytest.raises(ValueError, match=msg):
        parse_reduction_params({"reduction_distance": "canberra"}, is_knn=False)

    with pytest.raises(ValueError, match=msg):
        parse_reduction_params({"version": 3}, is_knn=True)


def test_parse_reduction_params_knn_rejects_reduction_aliases():
    """kNN rejects reduction-specific aliases (reduction_k, distance, voting)."""
    from models.knn.reduction import parse_reduction_params

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_k' for kNN"):
        parse_reduction_params({"reduction_k": 3}, is_knn=True)

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_distance' for kNN"):
        parse_reduction_params({"reduction_distance": "canberra"}, is_knn=True)

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_voting' for kNN"):
        parse_reduction_params({"reduction_voting": "inverse_distance"}, is_knn=True)


def test_parse_reduction_params_non_knn_accepts_distance_and_voting():
    """Non-kNN models accept reduction_distance and reduction_voting."""
    from models.knn.reduction import parse_reduction_params

    red = parse_reduction_params(
        {
            "reduction": "drop",
            "version": 3,
            "reduction_k": 5,
            "reduction_distance": "canberra",
            "reduction_voting": "inverse_distance",
        },
        is_knn=False,
    )
    assert red.method == "drop"
    assert red.k == 5
    assert red.distance == "canberra"
    assert red.voting == "inverse_distance"


def test_oop_subclasses_inheritance_and_override(synthetic_clusters):
    """Each reduction technique is a subclass of InstanceReduction and overrides reduce()."""
    subclasses = [
        (NoReduction(), "none"),
        (FCNNReduction(version=1), "fcnn"),
        (DROPReduction(version=3), "drop"),
        (ENNTHReduction(threshold=0.5), "ennth"),
        (DELReduction(k=3), "del"),
    ]
    for instance, expected_method in subclasses:
        assert isinstance(instance, InstanceReduction)
        assert instance.method == expected_method
        res = instance.reduce(synthetic_clusters)
        assert isinstance(res, ReductionResult)
        assert res.n_retained > 0
        assert res.n_retained <= res.n_original


def test_reduce_dataset_function(multiclass_data):
    """reduce_dataset helper provides single entrypoint for any method."""
    res = reduce_dataset(multiclass_data, method="fcnn", distance="manhattan")
    assert isinstance(res, ReductionResult)
    assert res.n_retained > 0


def test_reduction_returns_preprocessed_object(synthetic_clusters):
    """Reduction results contain a valid filtered Preprocessed object."""
    res_fcnn = fcnn(synthetic_clusters)
    assert isinstance(res_fcnn.data, Preprocessed)
    assert len(res_fcnn.data.y_codes) == res_fcnn.n_retained

    res_ennth = ennth(synthetic_clusters, threshold=0.6)
    assert isinstance(res_ennth.data, Preprocessed)
    assert len(res_ennth.data.y_codes) == res_ennth.n_retained

    res_drop = drop(synthetic_clusters, version=3)
    assert isinstance(res_drop.data, Preprocessed)
    assert len(res_drop.data.y_codes) == res_drop.n_retained

    res_del = del_reduction(synthetic_clusters, k=3)
    assert isinstance(res_del.data, Preprocessed)
    assert len(res_del.data.y_codes) == res_del.n_retained


def test_reduction_with_different_distances(multiclass_data):
    """Reduction techniques support various metrics (manhattan, canberra, callable)."""
    res_man = fcnn(multiclass_data, distance=manhattan)
    res_can = ennth(multiclass_data, distance="canberra")
    res_drop = drop(multiclass_data, version=3, distance="manhattan")
    res_del = del_reduction(multiclass_data, distance=manhattan)

    assert res_man.n_retained > 0
    assert res_can.n_retained > 0
    assert res_drop.n_retained > 0
    assert res_del.n_retained > 0


def test_reduction_registry():
    """REDUCTION_TECHNIQUES contains only expected methods."""
    assert set(REDUCTION_TECHNIQUES.keys()) >= {"fcnn", "ennth", "drop", "del"}


# ---------------------------------------------------------------------------
# Validation and Error Handling
# ---------------------------------------------------------------------------


def test_invalid_parameters():
    """Check validation of invalid configuration parameters."""
    dummy = make_preprocessed(np.ones((10, 2)), np.zeros(10))
    empty = make_preprocessed(np.empty((0, 2)), np.empty(0))

    with pytest.raises(ValueError, match="Unknown distance"):
        fcnn(dummy, distance="nonexistent_distance")

    with pytest.raises(ValueError, match="Unknown voting"):
        drop(dummy, voting="invalid_voting")

    with pytest.raises(ValueError, match="FCNN version must be 1, 2, 3, or 4"):
        fcnn(dummy, version=9)

    with pytest.raises(ValueError, match="version must be 1, 2, 3, 4, or 5"):
        drop(dummy, version=9)

    with pytest.raises(ValueError, match="must not be empty"):
        fcnn(empty)

    with pytest.raises(ValueError, match="positive integer"):
        ennth(dummy, k=-1)


def test_ennth_rejects_training_data_without_every_class():
    """A class of the header with no training case is a clear error, not an IndexError later."""
    data = make_preprocessed([[0.0], [0.1], [0.9], [1.0]], [0, 0, 2, 2])
    data = dataclasses.replace(data, class_values=("a", "b", "c"))
    with pytest.raises(ValueError, match=r"every class.*\['b'\]"):
        ennth(data, k=1)


# ---------------------------------------------------------------------------
# KNN Pipeline Integration & Correctness Verification Examples
# ---------------------------------------------------------------------------


def test_knn_example_fcnn_correctness(synthetic_clusters):
    """Example verifying KNN with FCNN reduction: reduces storage while keeping consistency."""
    knn = KNN({"k": 3, "reduction": "fcnn", "version": 1})
    result = knn.fit_predict(synthetic_clusters, synthetic_clusters)

    # Reduction: storage must be strictly less than original dataset size
    assert result.storage < len(synthetic_clusters.y_codes)
    # Accuracy on separable clusters remains high (> 90%)
    acc = np.mean(np.array(result.predictions) == synthetic_clusters.y_codes)
    assert acc > 0.90
    assert len(result.predictions) == len(synthetic_clusters.y_codes)


def test_knn_example_ennth_correctness(synthetic_clusters):
    """Example verifying KNN with ENNTH reduction: prunes noisy border instances."""
    knn = KNN({"k": 3, "reduction": "ennth", "threshold": 0.6})
    result = knn.fit_predict(synthetic_clusters, synthetic_clusters)

    # Outliers removed, clean instances retained
    assert result.storage <= len(synthetic_clusters.y_codes)
    acc = np.mean(np.array(result.predictions) == synthetic_clusters.y_codes)
    assert acc > 0.90


def test_knn_example_drop_correctness(synthetic_clusters):
    """Example verifying KNN with DROP reduction: high compression and accurate predictions."""
    knn = KNN({"k": 3, "reduction": "drop", "version": 3})
    result = knn.fit_predict(synthetic_clusters, synthetic_clusters)

    # DROP removes internal and noisy instances, cutting storage by > 30%
    assert result.storage < len(synthetic_clusters.y_codes) * 0.70
    acc = np.mean(np.array(result.predictions) == synthetic_clusters.y_codes)
    assert acc > 0.90


def test_knn_example_del_correctness(synthetic_clusters):
    """Example verifying KNN with DEL reduction: encoding length heuristic reduces instances."""
    knn = KNN({"k": 3, "reduction": "del"})
    result = knn.fit_predict(synthetic_clusters, synthetic_clusters)

    # DEL removes instances while preserving cost and boundary accuracy
    assert result.storage < len(synthetic_clusters.y_codes)
    acc = np.mean(np.array(result.predictions) == synthetic_clusters.y_codes)
    assert acc > 0.90


def test_knn_example_reduction_and_retention(synthetic_clusters):
    """Example verifying KNN with both reduction and retention enabled."""
    knn = KNN(
        {
            "k": 3,
            "reduction": "drop",
            "version": 3,
            "retention": "different_class",
        }
    )
    result = knn.fit_predict(synthetic_clusters, synthetic_clusters)
    assert len(result.predictions) == len(synthetic_clusters.y_codes)
    assert result.storage > 0
