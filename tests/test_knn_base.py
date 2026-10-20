"""Tests for the core k-nearest neighbours algorithm."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

import config
from models.knn.knn_base import (
    Neighbor,
    check_k,
    kNNAlgorithm,
    nearest_neighbors,
    predict,
)
from preprocessing import preprocess_fold

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeDataset:
    """Minimal dataset compatible with knn_base: one numeric column, no nominal ones."""

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
    """Plain-Python kNN: one distance per pair, neighbours sorted by (distance, index)."""
    predictions = []
    for query in test_rows:
        candidates = sorted(
            (pair_distance(query, row), index) for index, row in enumerate(train_rows)
        )
        nearest = [
            Neighbor(index=index, distance=distance, label=int(train_y[index]))
            for distance, index in candidates[:k]
        ]
        predictions.append(voting_function(nearest))
    return predictions


def manhattan_overlap_pair(a, b) -> float:
    """Sum of |numeric difference| plus the number of different nominal values."""
    (num_a, nom_a), (num_b, nom_b) = a, b
    numeric = sum(abs(float(x) - float(y)) for x, y in zip(num_a, num_b, strict=True))
    nominal = sum(x != y for x, y in zip(nom_a, nom_b, strict=True))
    return float(numeric + nominal)


def manhattan_overlap_matrix(train, test) -> np.ndarray:
    """Same distance as ``manhattan_overlap_pair`` for every (test, train) pair."""
    numeric = np.abs(test.X_num[:, None, :] - train.X_num[None, :, :]).sum(axis=2)
    nominal = (test.X_nom[:, None, :] != train.X_nom[None, :, :]).sum(axis=2)
    return numeric + nominal


def rows(dataset):
    return list(zip(dataset.X_num, dataset.X_nom, strict=True))


# ---------------------------------------------------------------------------
# Neighbor and check_k
# ---------------------------------------------------------------------------


def test_neighbor_stores_index_distance_and_label():
    neighbor = Neighbor(index=2, distance=0.5, label=1)

    assert neighbor.index == 2
    assert neighbor.distance == 0.5
    assert neighbor.label == 1


def test_check_k_accepts_python_and_numpy_integers():
    assert check_k(3) == 3
    assert check_k(np.int64(5)) == 5
    assert type(check_k(np.int64(5))) is int


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3"])
def test_check_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        check_k(k)


def test_check_k_rejects_k_above_the_limit():
    with pytest.raises(ValueError, match="k cannot exceed 2"):
        check_k(3, limit=2)


# ---------------------------------------------------------------------------
# nearest_neighbors
# ---------------------------------------------------------------------------


def test_nearest_neighbors_returns_k_nearest_instances():
    train = make_dataset([1, 5, 3, 8], labels=[0, 1, 1, 0])
    test = make_dataset([4])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)

    assert neighbors.indices.tolist() == [[1, 2]]
    assert neighbors.distances.tolist() == [[1, 1]]
    assert neighbors.labels.tolist() == [[1, 1]]


def test_nearest_neighbors_orders_each_test_case_by_distance():
    train = make_dataset([10, 2, 7, 4])
    test = make_dataset([3, 9])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=4)

    assert neighbors.distances.tolist() == [[1, 1, 4, 7], [1, 2, 5, 7]]


def test_nearest_neighbors_breaks_distance_ties_by_index():
    train = make_dataset([2, 4, 1, 5])
    test = make_dataset([3])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=4)

    assert neighbors.indices.tolist() == [[0, 1, 2, 3]]


def test_nearest_neighbors_keeps_lowest_index_at_the_k_boundary():
    train = make_dataset([2, 4, 7])
    test = make_dataset([3])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=1)

    assert neighbors.indices.tolist() == [[0]]


def test_nearest_neighbors_with_k_equal_to_training_size():
    train = make_dataset([1, 4, 7])
    test = make_dataset([3])

    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=3)

    assert neighbors.indices.shape == (1, 3)


def test_nearest_neighbors_rejects_empty_training_data():
    with pytest.raises(ValueError, match="training data must not be empty"):
        nearest_neighbors(make_dataset([]), make_dataset([1]), absolute_distance, k_max=1)


def test_nearest_neighbors_rejects_k_larger_than_training_size():
    with pytest.raises(ValueError, match="k cannot exceed"):
        nearest_neighbors(make_dataset([1, 2]), make_dataset([1]), absolute_distance, k_max=3)


def test_nearest_neighbors_rejects_a_distance_matrix_with_the_wrong_shape():
    def transposed(train, test):
        return absolute_distance(train, test).T

    with pytest.raises(ValueError, match="must return shape"):
        nearest_neighbors(make_dataset([1, 2]), make_dataset([1]), transposed, k_max=1)


@pytest.mark.parametrize("invalid_distance", [-1.0, float("inf"), float("nan")])
def test_nearest_neighbors_rejects_invalid_distances(invalid_distance):
    def invalid_distance_function(train, test):
        return np.full((len(test.y_codes), len(train.y_codes)), invalid_distance)

    with pytest.raises(ValueError, match="finite, non-negative"):
        nearest_neighbors(
            make_dataset([1, 2]), make_dataset([1]), invalid_distance_function, k_max=1
        )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_reuses_the_neighbours_for_every_smaller_k():
    train = make_dataset([1, 2, 3, 10, 11], labels=[0, 1, 1, 0, 0])
    test = make_dataset([1.4, 10.4])
    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=5)

    for k in (1, 3, 5):
        expected = kNNAlgorithm(train, test, k, absolute_distance, majority_vote)
        assert predict(neighbors, k, majority_vote) == expected


def test_predict_passes_neighbours_as_python_values():
    train = make_dataset([1, 5], labels=[0, 1])
    test = make_dataset([4])
    neighbors = nearest_neighbors(train, test, absolute_distance, k_max=2)
    received = []

    def recording_vote(nearest):
        received.extend(nearest)
        return nearest[0].label

    predict(neighbors, 2, recording_vote)

    assert received == [Neighbor(index=1, distance=1.0, label=1), Neighbor(0, 3.0, 0)]
    assert all(type(n.index) is int and type(n.label) is int for n in received)


def test_predict_rejects_k_above_the_computed_neighbours():
    neighbors = nearest_neighbors(
        make_dataset([1, 2, 3]), make_dataset([1]), absolute_distance, k_max=2
    )

    with pytest.raises(ValueError, match="k cannot exceed 2"):
        predict(neighbors, 3, first_neighbor_label)


@pytest.mark.parametrize("vote", [1.5, True, "1"])
def test_predict_rejects_a_non_integer_vote(vote):
    neighbors = nearest_neighbors(
        make_dataset([1, 2]), make_dataset([1]), absolute_distance, k_max=1
    )

    with pytest.raises(ValueError, match="integer class code"):
        predict(neighbors, 1, lambda nearest: vote)


# ---------------------------------------------------------------------------
# kNNAlgorithm
# ---------------------------------------------------------------------------


def test_kNNAlgorithm_predicts_using_injected_distance_and_voting():
    train = make_dataset([1, 5, 9], labels=[0, 1, 1])
    test = make_dataset([2, 8])

    predictions = kNNAlgorithm(train, test, 1, absolute_distance, first_neighbor_label)

    assert predictions == [0, 1]


def test_kNNAlgorithm_rejects_k_larger_than_training_size():
    with pytest.raises(ValueError, match="k cannot exceed"):
        kNNAlgorithm(
            make_dataset([1, 2]), make_dataset([1]), 3, absolute_distance, first_neighbor_label
        )


def test_kNNAlgorithm_matches_the_plain_python_reference():
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
        assert kNNAlgorithm(train, test, k, manhattan_overlap_matrix, majority_vote) == expected


def test_kNNAlgorithm_matches_the_plain_python_reference_on_a_real_fold():
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
