"""Tests for the KNN classifier adapter, the distance registry and the voting registry."""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import config
from distances import DISTANCES, euclidean
from experiment_runner import ClassificationResult
from knn_base import Neighbor, kNNAlgorithm
from models.knn import KNN, default_params
from preprocessing import Preprocessed, preprocess_fold
from voting import VOTING, majority


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
    assert all(isinstance(v, int | str) for v in KNN({}).params.values())


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
