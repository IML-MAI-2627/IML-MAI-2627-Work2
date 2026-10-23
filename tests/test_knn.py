"""Tests for the KNN classifier adapter, the distance registry and the voting registry."""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import config
from experiment_runner import ClassificationResult
from models.knn.distances import DISTANCES, canberra, clark, euclidean, heom, manhattan
from models.knn.knn import KNN, default_params
from models.knn.knn_utils import Neighbor, kNNAlgorithm, run_kNN
from models.knn.retention import (
    RETENTION,
    OblivionByGoodness,
    always_retain,
    different_class,
    never_retain,
)
from models.knn.voting import VOTING, inverse_distance, majority
from preprocessing import Preprocessed, preprocess_fold


def make_preprocessed(X_num, X_nom, labels, nom_values=None) -> Preprocessed:
    """Small real ``Preprocessed`` object (sklearn is used only as a test oracle here)."""
    X_num = np.asarray(X_num, dtype=float)  # 2-D, possibly with 0 columns
    X_nom = np.asarray(X_nom, dtype=np.int64)
    if nom_values is None:
        nom_values = tuple(
            tuple(str(v) for v in range(int(X_nom[:, j].max()) + 1)) for j in range(X_nom.shape[1])
        )
    class_values = tuple(str(c) for c in range(int(max(labels)) + 1))
    return Preprocessed(
        X_num=X_num,
        X_nom=X_nom,
        y=np.array([class_values[c] for c in labels]),
        y_codes=np.asarray(labels, dtype=np.int64),
        num_names=tuple(f"n{j}" for j in range(X_num.shape[1])),
        nom_names=tuple(f"c{j}" for j in range(X_nom.shape[1])),
        nom_values=nom_values,
        class_values=class_values,
    )


@pytest.fixture
def small_fold():
    train = make_preprocessed(
        X_num=[[0.0], [0.1], [0.9], [1.0]],
        X_nom=[[0], [0], [1], [1]],
        labels=[0, 0, 1, 1],
        nom_values=(("a", "b"),),
    )
    test = make_preprocessed(
        X_num=[[0.05], [0.95]],
        X_nom=[[0], [1]],
        labels=[0, 1],
        nom_values=(("a", "b"),),
    )
    return train, test


# ---------------------------------------------------------------------------
# parse_params
# ---------------------------------------------------------------------------


def test_parse_params_injects_defaults():
    assert KNN({}).params == default_params


def test_parse_params_casts_k_from_cli_string():
    assert KNN({"k": "5"}).params["k"] == 5


@pytest.mark.parametrize("k", ["None", "0", "-1", "abc", "1.5", "True", 0, True])
def test_parse_params_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        KNN({"k": k})


def test_parse_params_names_are_case_insensitive():
    knn = KNN({"distance": "EUCLIDEAN", "voting": "Majority"})
    assert knn.params["distance"] == "euclidean"
    assert knn.params["voting"] == "majority"


@pytest.mark.parametrize(
    "params, match",
    [
        ({"distance": "patata"}, "Invalid distance"),
        ({"voting": "patata"}, "Invalid voting"),
        ({"kk": 3}, "Unknown kNN parameters"),
    ],
)
def test_parse_params_rejects_unknown_values(params, match):
    with pytest.raises(ValueError, match=match):
        KNN(params)


def test_params_contain_only_plain_values():
    """Callables would be written as '<function ...>' in the CSV ``params`` column."""
    assert all(isinstance(v, int | float | str) for v in KNN({}).params.values())


def test_parse_params_reduction_adds_exclusive_parameters():
    # FCNN: adds version (default 1)
    knn_fcnn = KNN({"reduction": "fcnn"})
    assert knn_fcnn.params["reduction"] == "fcnn"
    assert knn_fcnn.params["version"] == 1
    assert "threshold" not in knn_fcnn.params

    # DROP: adds version (default 3)
    knn_drop = KNN({"reduction": "drop", "version": 4})
    assert knn_drop.params["reduction"] == "drop"
    assert knn_drop.params["version"] == 4
    assert "threshold" not in knn_drop.params

    # EENTh: adds threshold (default 0.0)
    knn_eenth = KNN({"reduction": "eenth", "threshold": 0.6})
    assert knn_eenth.params["reduction"] == "eenth"
    assert knn_eenth.params["threshold"] == 0.6
    assert "version" not in knn_eenth.params

    # DEL: neither version nor threshold
    knn_del = KNN({"reduction": "del"})
    assert knn_del.params["reduction"] == "del"
    assert "version" not in knn_del.params
    assert "threshold" not in knn_del.params

    # None: neither version nor threshold
    knn_none = KNN({"reduction": "none"})
    assert knn_none.params["reduction"] == "none"
    assert "version" not in knn_none.params
    assert "threshold" not in knn_none.params


def test_parse_params_reduction_rejects_incompatible_parameters():
    with pytest.raises(ValueError, match="Unknown parameter 'threshold' for FCNN"):
        KNN({"reduction": "fcnn", "threshold": 0.5})

    with pytest.raises(ValueError, match="Unknown parameter 'threshold' for DROP"):
        KNN({"reduction": "drop", "threshold": 0.5})

    with pytest.raises(ValueError, match="Unknown parameter 'version' for EENTh"):
        KNN({"reduction": "eenth", "version": 2})

    with pytest.raises(ValueError, match="Parameters.*are not allowed for DEL"):
        KNN({"reduction": "del", "version": 2})

    with pytest.raises(ValueError, match="Parameters.*are not allowed for DEL"):
        KNN({"reduction": "del", "threshold": 0.5})

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_k' for kNN"):
        KNN({"reduction": "del", "reduction_k": 3})

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_distance' for kNN"):
        KNN({"reduction": "drop", "reduction_distance": "canberra"})

    with pytest.raises(ValueError, match="Unknown parameter 'reduction_voting' for kNN"):
        KNN({"reduction": "drop", "reduction_voting": "inverse_distance"})

    err_none = "are not allowed when reduction is 'none'"
    with pytest.raises(ValueError, match=err_none):
        KNN({"reduction": "none", "version": 3})

    err_no_red = "without specifying the reduction type parameter '--reduction'"
    with pytest.raises(ValueError, match=err_no_red):
        KNN({"version": 3})

    with pytest.raises(ValueError, match=err_no_red):
        KNN({"threshold": 0.5})


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------


def test_fit_predict_rejects_k_larger_than_training_size(small_fold):
    train, test = small_fold
    with pytest.raises(ValueError, match="k cannot exceed"):
        KNN({"k": 5}).fit_predict(train, test)


def test_prepare_data_rejects_empty_data(small_fold):
    train, test = small_fold
    empty = Preprocessed(
        **{**vars(test), **{f: getattr(test, f)[:0] for f in ("X_num", "X_nom", "y", "y_codes")}}
    )
    with pytest.raises(ValueError, match="Test data must not be empty"):
        KNN({"k": 1}).prepare_data(train, empty)


def test_prepare_data_rejects_misaligned_data(small_fold):
    train, test = small_fold
    broken = Preprocessed(**{**vars(test), "y_codes": test.y_codes[:1]})
    with pytest.raises(ValueError, match="same number of instances"):
        KNN({"k": 1}).prepare_data(train, broken)


def test_prepare_data_applies_reduction(small_fold):
    train, test = small_fold
    knn = KNN({"k": 1, "reduction": "drop", "version": 3})
    red_train, red_test = knn.prepare_data(train, test)
    assert len(red_train.y) <= len(train.y)
    assert len(red_test.y) == len(test.y)


# ---------------------------------------------------------------------------
# fit_predict
# ---------------------------------------------------------------------------


def test_fit_predict_returns_classification_result(small_fold):
    train, test = small_fold
    result = KNN({"k": 1}).fit_predict(train, test)

    assert isinstance(result, ClassificationResult)
    assert result.predictions == [0, 1]
    assert result.storage == len(train.y)


def test_fit_predict_matches_kNNAlgorithm(small_fold):
    train, test = small_fold
    expected = kNNAlgorithm(train, test, 3, euclidean, majority)
    assert KNN({"k": 3}).fit_predict(train, test).predictions == expected


def test_fit_predict_does_not_modify_the_folds(small_fold):
    train, test = small_fold
    before = [(d.X_num.copy(), d.X_nom.copy(), d.y_codes.copy()) for d in (train, test)]

    KNN({"k": 3}).fit_predict(train, test)

    for data, (X_num, X_nom, y_codes) in zip((train, test), before, strict=True):
        np.testing.assert_array_equal(data.X_num, X_num)
        np.testing.assert_array_equal(data.X_nom, X_nom)
        np.testing.assert_array_equal(data.y_codes, y_codes)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def test_registries_contain_the_defaults():
    assert default_params["distance"] in DISTANCES
    assert default_params["voting"] in VOTING


def test_euclidean_matches_hand_computed_values(small_fold):
    train, test = small_fold
    # One-hot rows: [num, a, b]. Different nominal value adds (1 + 1) to the squared sum.
    expected = np.sqrt(
        [
            [0.05**2, 0.05**2, 0.85**2 + 2, 0.95**2 + 2],
            [0.95**2 + 2, 0.85**2 + 2, 0.05**2, 0.05**2],
        ]
    )
    np.testing.assert_allclose(euclidean(train, test), expected)


def test_euclidean_gives_exact_zero_for_identical_rows(small_fold):
    train, _ = small_fold
    assert np.all(np.diag(euclidean(train, train)) == 0.0)


@pytest.mark.parametrize("n_num", [0, 3])  # 0 numeric columns, as in splice
def test_euclidean_matches_direct_one_hot_computation(n_num):
    rng = np.random.default_rng(config.SEED)
    nom_values = (("a", "b", "c"), ("x", "y"), ("p", "q", "r", "s"))

    def random_fold(n):
        X_nom = np.column_stack([rng.integers(0, len(v), n) for v in nom_values])
        return make_preprocessed(
            rng.random((n, n_num)), X_nom, rng.integers(0, 2, n).tolist(), nom_values
        )

    train, test = random_fold(30), random_fold(10)
    direct = np.sqrt(
        ((test.to_onehot()[:, None, :] - train.to_onehot()[None, :, :]) ** 2).sum(axis=2)
    )
    np.testing.assert_allclose(euclidean(train, test), direct, rtol=0, atol=1e-12)


def test_manhattan_matches_hand_computed_values(small_fold):
    train, test = small_fold
    # |numeric difference| + 2 * (different nominal values), see the table in the PR.
    expected = np.array(
        [
            [0.05, 0.05, 2.85, 2.95],
            [2.95, 2.85, 0.05, 0.05],
        ]
    )
    np.testing.assert_allclose(manhattan(train, test), expected)


def test_manhattan_gives_exact_zero_for_identical_rows(small_fold):
    train, _ = small_fold
    assert np.all(np.diag(manhattan(train, train)) == 0.0)


@pytest.mark.parametrize("n_num", [0, 3])
def test_manhattan_matches_direct_one_hot_computation(n_num):
    # Sum of |differences| over the full one-hot vectors, computed the slow, obvious way.
    rng = np.random.default_rng(0)
    nom_values = (("a", "b", "c"), ("x", "y", "z"))
    train = make_preprocessed(
        rng.random((8, n_num)), rng.integers(0, 3, size=(8, 2)), [0] * 8, nom_values
    )
    test = make_preprocessed(
        rng.random((5, n_num)), rng.integers(0, 3, size=(5, 2)), [0] * 5, nom_values
    )
    a, b = test.to_onehot(), train.to_onehot()
    direct = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    np.testing.assert_allclose(manhattan(train, test), direct, rtol=0, atol=1e-12)


def test_manhattan_is_registered():
    assert DISTANCES["manhattan"] is manhattan


def test_canberra_matches_hand_computed_values(small_fold):
    train, test = small_fold
    # sum |x - y| / (|x| + |y|) over numeric values + 2 * (different nominal values).
    expected = np.array(
        [
            [1.0, 0.05 / 0.15, 0.85 / 0.95 + 2, 0.95 / 1.05 + 2],
            [3.0, 0.85 / 1.05 + 2, 0.05 / 1.85, 0.05 / 1.95],
        ]
    )
    np.testing.assert_allclose(canberra(train, test), expected)


def test_canberra_treats_zero_over_zero_as_zero():
    # Both numeric values are 0 -> 0/0, which must count as 0 (no nan, no warning).
    train = make_preprocessed([[0.0]], [[0]], [0], (("a", "b"),))
    test = make_preprocessed([[0.0]], [[0]], [0], (("a", "b"),))
    assert canberra(train, test)[0, 0] == 0.0


def test_canberra_gives_exact_zero_for_identical_rows(small_fold):
    train, _ = small_fold
    assert np.all(np.diag(canberra(train, train)) == 0.0)


@pytest.mark.parametrize("n_num", [0, 3])
def test_canberra_matches_direct_one_hot_computation(n_num):
    rng = np.random.default_rng(0)
    nom_values = (("a", "b", "c"), ("x", "y", "z"))
    train = make_preprocessed(
        rng.random((8, n_num)), rng.integers(0, 3, size=(8, 2)), [0] * 8, nom_values
    )
    test = make_preprocessed(
        rng.random((5, n_num)), rng.integers(0, 3, size=(5, 2)), [0] * 5, nom_values
    )
    a, b = test.to_onehot(), train.to_onehot()
    direct = np.zeros((5, 8))
    for i in range(5):
        for j in range(8):
            for x, y in zip(a[i], b[j], strict=True):
                if x + y > 0:
                    direct[i, j] += abs(x - y) / (abs(x) + abs(y))
    np.testing.assert_allclose(canberra(train, test), direct, rtol=0, atol=1e-12)


def test_canberra_is_registered():
    assert DISTANCES["canberra"] is canberra


def test_clark_matches_hand_computed_values(small_fold):
    train, test = small_fold
    # sum (|x - y| / (|x| + |y|))^2 over numeric values + 2 * (different nominal values).
    expected = np.array(
        [
            [1.0, (0.05 / 0.15) ** 2, (0.85 / 0.95) ** 2 + 2, (0.95 / 1.05) ** 2 + 2],
            [3.0, (0.85 / 1.05) ** 2 + 2, (0.05 / 1.85) ** 2, (0.05 / 1.95) ** 2],
        ]
    )
    np.testing.assert_allclose(clark(train, test), expected)


def test_clark_treats_zero_over_zero_as_zero():
    train = make_preprocessed([[0.0]], [[0]], [0], (("a", "b"),))
    test = make_preprocessed([[0.0]], [[0]], [0], (("a", "b"),))
    assert clark(train, test)[0, 0] == 0.0


def test_clark_gives_exact_zero_for_identical_rows(small_fold):
    train, _ = small_fold
    assert np.all(np.diag(clark(train, train)) == 0.0)


def test_clark_has_same_neighbour_order_as_its_square_root_version(small_fold):
    # The square root is monotonic, so it cannot change which neighbours are nearest.
    train, test = small_fold
    d = clark(train, test)
    np.testing.assert_array_equal(
        np.argsort(d, axis=1, kind="stable"), np.argsort(np.sqrt(d), axis=1, kind="stable")
    )


def test_clark_is_registered():
    assert DISTANCES["clark"] is clark


def test_heom_matches_hand_computed_values(small_fold):
    train, test = small_fold
    # sqrt(numeric difference^2 + number of different nominal values).
    expected = np.array(
        [
            [0.05, 0.05, np.sqrt(0.85**2 + 1), np.sqrt(0.95**2 + 1)],
            [np.sqrt(0.95**2 + 1), np.sqrt(0.85**2 + 1), 0.05, 0.05],
        ]
    )
    np.testing.assert_allclose(heom(train, test), expected)


def test_heom_gives_exact_zero_for_identical_rows(small_fold):
    train, _ = small_fold
    assert np.all(np.diag(heom(train, train)) == 0.0)


def test_heom_has_same_neighbour_order_as_euclidean_without_numeric_attributes():
    # Only nominal attributes (like splice): heom = sqrt(m), euclidean = sqrt(2m).
    rng = np.random.default_rng(0)
    nom_values = (("a", "b", "c"),) * 4
    train = make_preprocessed(
        np.zeros((8, 0)), rng.integers(0, 3, size=(8, 4)), [0] * 8, nom_values
    )
    test = make_preprocessed(np.zeros((5, 0)), rng.integers(0, 3, size=(5, 4)), [0] * 5, nom_values)
    np.testing.assert_allclose(heom(train, test) * np.sqrt(2), euclidean(train, test))


def test_heom_is_registered():
    assert DISTANCES["heom"] is heom


@pytest.mark.parametrize("metric_name", ["euclidean", "manhattan", "canberra", "clark", "heom"])
def test_distance_single_argument_matches_symmetric(small_fold, metric_name):
    train, _ = small_fold
    fn = DISTANCES[metric_name]
    # Single argument call using triu_indices
    d_single = fn(train)
    # Two argument call
    d_double = fn(train, train)
    np.testing.assert_allclose(d_single, d_double)
    # Diagonal is strictly zero
    assert np.all(np.diag(d_single) == 0.0)
    # Exact symmetry
    np.testing.assert_allclose(d_single, d_single.T)


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def test_majority_picks_the_most_frequent_class():
    neighbors = [Neighbor(0, 0.1, 2), Neighbor(1, 0.2, 1), Neighbor(2, 0.3, 1)]
    assert majority(neighbors) == 1


def as_neighbors(labels):
    """Neighbours in (distance, index) order with the given labels."""
    return [Neighbor(index=i, distance=0.1 * (i + 1), label=c) for i, c in enumerate(labels)]


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([2, 0, 0, 2], 2),  # nearest tied class has the larger code
        ([0, 2, 2, 0], 0),  # nearest tied class has the smaller code
        ([1, 2, 0, 0, 2], 2),  # nearest neighbour is not in a tied class
    ],
)
def test_majority_tie_goes_to_the_nearest_tied_class(labels, expected):
    assert majority(as_neighbors(labels)) == expected


def test_inverse_distance_matches_hand_computed_vote():
    # Class 0: 1/1 = 1.0. Class 1: 1/4 + 1/4 = 0.5. Class 0 wins, although majority says 1.
    neighbors = [Neighbor(0, 1.0, 0), Neighbor(1, 4.0, 1), Neighbor(2, 4.0, 1)]
    assert inverse_distance(neighbors) == 0
    assert majority(neighbors) == 1


def test_inverse_distance_tie_goes_to_nearest_neighbour():
    # Class 0: 1/1 = 1.0. Class 1: 1/2 + 1/2 = 1.0. Exact tie -> class of the nearest (0).
    neighbors = [Neighbor(0, 1.0, 0), Neighbor(1, 2.0, 1), Neighbor(2, 2.0, 1)]
    assert inverse_distance(neighbors) == 0


def test_inverse_distance_only_zero_distance_neighbours_vote():
    # One identical case (d=0) beats two very close cases of another class.
    neighbors = [Neighbor(0, 0.0, 1), Neighbor(1, 0.5, 0), Neighbor(2, 0.5, 0)]
    assert inverse_distance(neighbors) == 1


def test_inverse_distance_tie_between_zero_distance_neighbours_goes_to_first():
    # Two identical cases of different classes: equal weight, the first (lowest index) wins.
    neighbors = [Neighbor(3, 0.0, 1), Neighbor(5, 0.0, 0), Neighbor(1, 0.1, 0)]
    assert inverse_distance(neighbors) == 1


def test_inverse_distance_is_registered():
    assert VOTING["inverse_distance"] is inverse_distance


# ---------------------------------------------------------------------------
# Sanity check against scikit-learn (tests only, never in deliverable code)
# ---------------------------------------------------------------------------


def test_k1_matches_sklearn_on_one_hot_data():
    rng = np.random.default_rng(config.SEED)
    train = make_preprocessed(
        rng.random((60, 3)), rng.integers(0, 3, (60, 2)), rng.integers(0, 3, 60).tolist()
    )
    test = make_preprocessed(
        rng.random((20, 3)), rng.integers(0, 3, (20, 2)), rng.integers(0, 3, 20).tolist()
    )
    test = Preprocessed(**{**vars(test), "nom_values": train.nom_values})

    reference = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    reference.fit(train.to_onehot(), train.y_codes)

    predictions = KNN({"k": 1}).fit_predict(train, test).predictions
    assert predictions == reference.predict(test.to_onehot()).tolist()


def test_real_fold_accuracy_is_far_above_chance():
    train_path, test_path = config.fold_paths("vowel", 0)
    if not (train_path.exists() and test_path.exists()):
        pytest.skip("ARFF files not in datasets/vowel/")
    train, test, _ = preprocess_fold("vowel", 0)

    predictions = KNN({"k": 1}).fit_predict(train, test).predictions
    assert np.mean(np.asarray(predictions) == test.y_codes) > 0.9


@pytest.fixture
def retention_fold():
    # 1-D, k=1. Train: x=0 (class 0), x=1 (class 1). Test: x=0.9 and x=0.85, both class 0.
    # Without retention both are nearest to x=1 -> predicted 1 (wrong).
    # Retaining the first one (x=0.9, class 0) makes it the nearest of the second -> 0.
    no_nom = np.zeros((2, 0), dtype=int)
    train = make_preprocessed([[0.0], [1.0]], no_nom, [0, 1])
    test = make_preprocessed([[0.9], [0.85]], no_nom, [0, 0])
    return train, test


def test_never_retain_predictions(retention_fold):
    train, test = retention_fold
    run = run_kNN(train, test, 1, euclidean, majority, never_retain)
    assert run.predictions == [1, 1]
    assert run.case_base_size == 2


def test_always_retain_uses_earlier_test_cases(retention_fold):
    train, test = retention_fold
    run = run_kNN(train, test, 1, euclidean, majority, always_retain)
    assert run.predictions == [1, 0]
    assert run.case_base_size == 4  # 2 train + both test cases


def test_different_class_retains_only_mistakes(retention_fold):
    train, test = retention_fold
    run = run_kNN(train, test, 1, euclidean, majority, different_class)
    assert run.predictions == [1, 0]
    assert run.case_base_size == 3  # 2 train + the first (wrong) one; the second was right


@pytest.mark.parametrize("k", [1, 3, 5])
def test_never_retain_matches_batch_path(k):
    # The one-by-one loop with NR must give exactly the fast batch predictions.
    rng = np.random.default_rng(0)
    nom_values = (("a", "b", "c"),)
    train = make_preprocessed(
        rng.random((20, 2)),
        rng.integers(0, 3, size=(20, 1)),
        rng.integers(0, 3, size=20),
        nom_values,
    )
    test = make_preprocessed(
        rng.random((15, 2)),
        rng.integers(0, 3, size=(15, 1)),
        rng.integers(0, 3, size=15),
        nom_values,
    )
    sequential = run_kNN(train, test, k, euclidean, majority, never_retain)
    assert sequential.predictions == run_kNN(train, test, k, euclidean, majority).predictions
    assert sequential.case_base_size == 20


def test_retention_registry():
    assert RETENTION == {
        "never": never_retain,
        "always": always_retain,
        "different_class": different_class,
        "oblivion_by_goodness": OblivionByGoodness(),
    }


def test_parse_params_resolves_retention_case_insensitively():
    assert KNN({"retention": "Always"}).params["retention"] == "always"


def test_parse_params_rejects_unknown_retention():
    with pytest.raises(ValueError, match="Invalid retention"):
        KNN({"retention": "sometimes"})


@pytest.mark.parametrize(
    ("retention", "expected_storage"),
    [("never", 4), ("always", 6)],  # small_fold: 4 train cases, 2 test cases
)
def test_fit_predict_storage_is_final_case_base_size(small_fold, retention, expected_storage):
    train, test = small_fold
    result = KNN({"k": 1, "retention": retention}).fit_predict(train, test)
    assert result.storage == expected_storage


@pytest.fixture
def og_fold():
    no_nom_train = np.zeros((2, 0), dtype=int)
    no_nom_test = np.zeros((3, 0), dtype=int)
    train = make_preprocessed([[0.0], [1.0]], no_nom_train, [0, 1])
    test = make_preprocessed([[0.2], [0.9], [0.95]], no_nom_test, [0, 0, 1])
    return train, test


def test_oblivion_by_goodness_hand_computed_sequence(og_fold):
    # q0 -> A right (G_A 0.6); q1 -> B wrong (G_B 0.4, forgotten);
    # q2 -> q1 wrong (G_q1 0.4, forgotten). Final case base: A, q0, q2.
    train, test = og_fold
    run = run_kNN(train, test, 1, euclidean, majority, OblivionByGoodness())
    assert run.predictions == [0, 1, 0]
    assert run.case_base_size == 3


def test_oblivion_by_goodness_forgets_compared_with_always_retain(og_fold):
    train, test = og_fold
    always = run_kNN(train, test, 1, euclidean, majority, always_retain)
    assert always.case_base_size == 5


def test_oblivion_by_goodness_never_leaves_fewer_than_k_cases():
    # k=2 and both training cases are wrong for q0: both drop to 0.4, but deleting them
    # would leave fewer than k cases, so they stay. q0 is then retained -> 3 cases.
    no_nom = np.zeros((2, 0), dtype=int)
    train = make_preprocessed([[0.0], [1.0]], no_nom, [0, 0])
    test = make_preprocessed([[0.5]], np.zeros((1, 0), dtype=int), [1])
    run = run_kNN(train, test, 2, euclidean, majority, OblivionByGoodness())
    assert run.case_base_size == 3
