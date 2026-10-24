"""Tests for core k-nearest neighbours utilities and algorithms."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

import config
from models.knn.knn_utils import (
    Neighbor,
    check_k,
    nearest_neighbors,
    predict,
    run_kNN,
)
from preprocessing import preprocess_fold

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeDataset:
    """Minimal dataset compatible with knn_utils: one numeric column, no nominal ones."""

    X_num: np.ndarray
    X_nom: np.ndarray
    y_codes: np.ndarray
    y: np.ndarray


def make_dataset(values, labels=None) -> FakeDataset:
    labels = [0] * len(values) if labels is None else labels
    return FakeDataset(
        X_num=np.array(values, dtype=float).reshape(-1, 1),
        X_nom=np.empty((len(values), 0), dtype=np.int64),
        y_codes=np.array(labels, dtype=np.int64),
        y=np.array(labels),
    )


def absolute_distance(train, test) -> np.ndarray:
    """(n_test, n_train) matrix of |test - train| on the single numeric column."""
    return np.abs(test.X_num[:, :1] - train.X_num[:, 0])


def first_neighbor_label(neighbors) -> int:
    """Test voting function that returns the label of the nearest neighbor."""
    return neighbors[0].label


def majority_vote(neighbors) -> int:
    """Most frequent label; ties go to the smallest class code."""
    counts = Counter(neighbor.label for neighbor in neighbors)
    return min(counts, key=lambda label: (-counts[label], label))


def reference_knn(train_rows, train_y, test_rows, k, pair_distance, voting_function):
    """Plain-Python kNN reference implementation used to double-check numpy."""
    predictions = []
    for test_row in test_rows:
        pairs = []
        for i, (train_row, label) in enumerate(zip(train_rows, train_y, strict=True)):
            pairs.append((pair_distance(train_row, test_row), i, label))
        pairs.sort(key=lambda t: (t[0], t[1]))
        nearest = [Neighbor(index=idx, distance=dist, label=lbl) for dist, idx, lbl in pairs[:k]]
        predictions.append(voting_function(nearest))
    return predictions


def manhattan_overlap_pair(a, b):
    num_diff = sum(abs(x - y) for x, y in zip(a[0], b[0], strict=True))
    nom_diff = sum(x != y for x, y in zip(a[1], b[1], strict=True))
    return num_diff + nom_diff


def manhattan_overlap_matrix(train, test):
    num = np.sum(np.abs(test.X_num[:, None, :] - train.X_num[None, :, :]), axis=2)
    nom = np.sum(test.X_nom[:, None, :] != train.X_nom[None, :, :], axis=2)
    return num + nom


def rows(data):
    return list(zip(data.X_num.tolist(), data.X_nom.tolist(), strict=True))


# ---------------------------------------------------------------------------
# Neighbor
# ---------------------------------------------------------------------------


def test_neighbor_stores_index_distance_and_label():
    n = Neighbor(index=3, distance=1.5, label=0)
    assert n.index == 3
    assert n.distance == 1.5
    assert n.label == 0


# ---------------------------------------------------------------------------
# check_k
# ---------------------------------------------------------------------------


def test_check_k_accepts_python_and_numpy_integers():
    assert check_k(1) == 1
    assert check_k(np.int64(5)) == 5
    assert type(check_k(np.int64(5))) is int


@pytest.mark.parametrize("k", [0, -1, 1.5, True, False, "3", None])
def test_check_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        check_k(k)


def test_check_k_rejects_k_above_the_limit():
    check_k(5, limit=5)
    with pytest.raises(ValueError, match="cannot exceed 5"):
        check_k(6, limit=5)


# ---------------------------------------------------------------------------
# nearest_neighbors
# ---------------------------------------------------------------------------


def test_nearest_neighbors_returns_k_nearest_instances():
    train = make_dataset([0, 10, 20])
    test = make_dataset([9])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    assert neighbors.indices.tolist() == [[1, 0]]
    assert neighbors.distances.tolist() == [[1.0, 9.0]]


def test_nearest_neighbors_orders_each_test_case_by_distance():
    train = make_dataset([0, 5, 10])
    test = make_dataset([4, 6])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    assert neighbors.indices.tolist() == [[1, 0], [1, 2]]
    assert neighbors.distances.tolist() == [[1.0, 4.0], [1.0, 4.0]]


def test_nearest_neighbors_breaks_distance_ties_by_index():
    train = make_dataset([0, 2])
    test = make_dataset([1])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    assert neighbors.indices.tolist() == [[0, 1]]


def test_nearest_neighbors_keeps_lowest_index_at_the_k_boundary():
    train = make_dataset([0, 2, 2])
    test = make_dataset([1])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    assert neighbors.indices.tolist() == [[0, 1]]


def test_nearest_neighbors_with_k_equal_to_training_size():
    train = make_dataset([3, 1, 2])
    test = make_dataset([0])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=3)

    assert neighbors.indices.tolist() == [[1, 2, 0]]
    assert neighbors.distances.tolist() == [[1.0, 2.0, 3.0]]


def test_nearest_neighbors_rejects_empty_training_data():
    with pytest.raises(ValueError, match="training data must not be empty"):
        nearest_neighbors(make_dataset([]), make_dataset([1]), absolute_distance, k_max=1)


def test_nearest_neighbors_rejects_k_larger_than_training_size():
    with pytest.raises(ValueError, match="k cannot exceed"):
        nearest_neighbors(make_dataset([1, 2]), make_dataset([1]), absolute_distance, k_max=3)


def test_nearest_neighbors_rejects_a_distance_matrix_with_the_wrong_shape():
    train = make_dataset([1, 2, 3])
    test = make_dataset([1, 2])

    def bad_distance(train, test):
        return np.zeros((len(test.y_codes), len(train.y_codes) - 1))

    with pytest.raises(ValueError, match=r"shape \(2, 3\), got \(2, 2\)"):
        nearest_neighbors(train, test, bad_distance, k_max=1)


@pytest.mark.parametrize("invalid_distance", [-1.0, float("inf"), float("nan")])
def test_nearest_neighbors_rejects_invalid_distances(invalid_distance):
    train = make_dataset([1])
    test = make_dataset([1])

    def bad_distance(train, test):
        return np.array([[invalid_distance]])

    with pytest.raises(ValueError, match="finite, non-negative"):
        nearest_neighbors(train, test, bad_distance, k_max=1)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_reuses_the_neighbours_for_every_smaller_k():
    train = make_dataset([0, 10, 20], labels=[0, 1, 0])
    test = make_dataset([9])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=3)

    for k in (1, 2, 3):
        expected = run_kNN(train, test, k, absolute_distance, majority_vote).predictions
        assert predict(neighbors, k, majority_vote) == expected


def test_predict_passes_neighbours_as_python_values():
    train = make_dataset([1.0], labels=[0])
    test = make_dataset([2.0])
    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=1)

    captured = []

    def capturing_voting(nearest):
        captured.extend(nearest)
        return 0

    predict(neighbors, 1, capturing_voting)

    assert len(captured) == 1
    assert type(captured[0].index) is int
    assert type(captured[0].distance) is float
    assert type(captured[0].label) is int


def test_predict_rejects_k_above_the_computed_neighbours():
    train = make_dataset([1, 2, 3])
    test = make_dataset([1])
    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    with pytest.raises(ValueError, match="k cannot exceed 2"):
        predict(neighbors, 3, first_neighbor_label)


@pytest.mark.parametrize("vote", [1.5, True, "1"])
def test_predict_rejects_a_non_integer_vote(vote):
    train = make_dataset([1])
    test = make_dataset([1])
    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=1)

    with pytest.raises(ValueError, match="integer class code"):
        predict(neighbors, 1, lambda nearest: vote)


# ---------------------------------------------------------------------------
# run_kNN
# ---------------------------------------------------------------------------


def test_run_kNN_predicts_using_injected_distance_and_voting():
    train = make_dataset([1, 5, 9], labels=[0, 1, 1])
    test = make_dataset([2, 8])

    predictions = run_kNN(train, test, 1, absolute_distance, first_neighbor_label).predictions

    assert predictions == [0, 1]


def test_run_kNN_rejects_k_larger_than_training_size():
    with pytest.raises(ValueError, match="k cannot exceed"):
        run_kNN(make_dataset([1, 2]), make_dataset([1]), 3, absolute_distance, first_neighbor_label)


def test_run_kNN_matches_the_plain_python_reference():
    """Many equal distances (integer features), so the tie-break is exercised too."""
    rng = np.random.default_rng(config.SEED)
    train = FakeDataset(
        X_num=rng.integers(0, 3, size=(40, 2)).astype(float),
        X_nom=rng.integers(0, 3, size=(40, 2)),
        y_codes=rng.integers(0, 3, size=40),
        y=np.zeros(40),
    )
    test = FakeDataset(
        X_num=rng.integers(0, 3, size=(15, 2)).astype(float),
        X_nom=rng.integers(0, 3, size=(15, 2)),
        y_codes=np.zeros(15, dtype=np.int64),
        y=np.zeros(15),
    )

    for k in (1, 3, 5, 7):
        expected = reference_knn(
            rows(train), train.y_codes, rows(test), k, manhattan_overlap_pair, majority_vote
        )
        actual = run_kNN(train, test, k, manhattan_overlap_matrix, majority_vote).predictions
        assert actual == expected


def test_run_kNN_matches_the_plain_python_reference_on_a_real_fold():
    train_path, test_path = config.fold_paths("vowel", 0)
    if not (train_path.exists() and test_path.exists()):
        pytest.skip("ARFF files not in datasets/vowel/")
    train, test, _ = preprocess_fold("vowel", 0)

    neighbors = nearest_neighbors(train, test, manhattan_overlap_matrix, k_max=7)
    for k in (3, 5, 7):
        expected = reference_knn(
            rows(train), train.y_codes, rows(test), k, manhattan_overlap_pair, majority_vote
        )
        assert predict(neighbors, k, majority_vote) == expected


def test_run_kNN_with_precomputed_distance_matrix():
    train = make_dataset([0, 10, 20], labels=[0, 1, 0])
    test = make_dataset([9, 21])
    D = absolute_distance(train, test)

    run_normal = run_kNN(train, test, 1, absolute_distance, majority_vote)
    run_precomputed = run_kNN(train, test, 1, absolute_distance, majority_vote, distance_matrix=D)
    assert run_precomputed.predictions == run_normal.predictions
    assert run_precomputed.case_base_size == run_normal.case_base_size
