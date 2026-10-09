"""Tests for the core k-nearest neighbours algorithm."""

from dataclasses import dataclass

import pytest

from knn_base import (
    Neighbor,
    find_neighbors,
    knn_algorithm,
    make_knn_classifier,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def absolute_distance(a, b) -> float:
    """Test distance function using one-dimensional numeric samples."""
    return abs(a - b)


def first_neighbor_label(neighbors) -> int:
    """Test voting function that returns the label of the nearest neighbor."""
    return neighbors[0].label


@dataclass
class FakeDataset:
    """Minimal dataset compatible with knn_algorithm."""

    X_num: list
    X_nom: list
    y_codes: list
    y: list


# ---------------------------------------------------------------------------
# Neighbor
# ---------------------------------------------------------------------------


def test_neighbor_stores_index_distance_and_label():
    neighbor = Neighbor(index=2, distance=0.5, label=1)

    assert neighbor.index == 2
    assert neighbor.distance == 0.5
    assert neighbor.label == 1


# ---------------------------------------------------------------------------
# find_neighbors
# ---------------------------------------------------------------------------


def test_find_neighbors_returns_k_nearest_instances():
    train_X = [1, 5, 3, 8]
    train_y = [0, 1, 1, 0]

    neighbors = find_neighbors(
        train_X=train_X,
        train_y=train_y,
        query=4,
        k=2,
        distance_function=absolute_distance,
    )

    assert [neighbor.index for neighbor in neighbors] == [1, 2]
    assert [neighbor.distance for neighbor in neighbors] == [1, 1]
    assert [neighbor.label for neighbor in neighbors] == [1, 1]


def test_find_neighbors_returns_neighbors_in_distance_order():
    train_X = [10, 2, 7, 4]
    train_y = [0, 1, 0, 1]

    neighbors = find_neighbors(
        train_X=train_X,
        train_y=train_y,
        query=3,
        k=4,
        distance_function=absolute_distance,
    )

    assert [neighbor.distance for neighbor in neighbors] == [1, 1, 4, 7]


def test_find_neighbors_breaks_distance_ties_by_index():
    train_X = [2, 4, 1, 5]
    train_y = [0, 1, 0, 1]

    neighbors = find_neighbors(
        train_X=train_X,
        train_y=train_y,
        query=3,
        k=4,
        distance_function=absolute_distance,
    )

    assert [neighbor.index for neighbor in neighbors] == [0, 1, 2, 3]


def test_find_neighbors_keeps_lowest_index_at_the_k_boundary():
    neighbors = find_neighbors(
        train_X=[2, 4, 7],
        train_y=[0, 1, 1],
        query=3,
        k=1,
        distance_function=absolute_distance,
    )

    assert [neighbor.index for neighbor in neighbors] == [0]


def test_find_neighbors_with_k_equal_to_training_size():
    train_X = [1, 4, 7]
    train_y = [0, 1, 0]

    neighbors = find_neighbors(
        train_X=train_X,
        train_y=train_y,
        query=3,
        k=3,
        distance_function=absolute_distance,
    )

    assert len(neighbors) == 3


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_find_neighbors_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        find_neighbors(
            train_X=[1, 2],
            train_y=[0, 1],
            query=1,
            k=k,
            distance_function=absolute_distance,
        )


def test_find_neighbors_rejects_empty_training_data():
    with pytest.raises(ValueError, match="training data must not be empty"):
        find_neighbors(
            train_X=[],
            train_y=[],
            query=1,
            k=1,
            distance_function=absolute_distance,
        )


def test_find_neighbors_rejects_mismatched_features_and_labels():
    with pytest.raises(ValueError, match="same number of instances"):
        find_neighbors(
            train_X=[1, 2],
            train_y=[0],
            query=1,
            k=1,
            distance_function=absolute_distance,
        )


def test_find_neighbors_rejects_k_larger_than_training_size():
    with pytest.raises(ValueError, match="k cannot exceed"):
        find_neighbors(
            train_X=[1, 2],
            train_y=[0, 1],
            query=1,
            k=3,
            distance_function=absolute_distance,
        )


@pytest.mark.parametrize("invalid_distance", [-1.0, float("inf"), float("nan")])
def test_find_neighbors_rejects_invalid_distances(invalid_distance):
    def invalid_distance_function(a, b):
        return invalid_distance

    with pytest.raises(ValueError, match="finite, non-negative"):
        find_neighbors(
            train_X=[1, 2],
            train_y=[0, 1],
            query=1,
            k=1,
            distance_function=invalid_distance_function,
        )


# ---------------------------------------------------------------------------
# knn_algorithm
# ---------------------------------------------------------------------------


def test_knn_algorithm_predicts_using_injected_distance_and_voting():
    train = FakeDataset(
        X_num=[[1], [5], [9]],
        X_nom=[[], [], []],
        y_codes=[0, 1, 1],
        y=[0, 1, 1],
    )
    test = FakeDataset(
        X_num=[[2], [8]],
        X_nom=[[], []],
        y_codes=[],
        y=[],
    )

    def numeric_distance(a, b):
        return abs(a[0][0] - b[0][0])

    predictions = knn_algorithm(
        train=train,
        test=test,
        k=1,
        distance_function=numeric_distance,
        voting_function=first_neighbor_label,
    )

    assert predictions == [0, 1]


def test_knn_algorithm_rejects_empty_training_data():
    train = FakeDataset(
        X_num=[],
        X_nom=[],
        y_codes=[],
        y=[],
    )
    test = FakeDataset(
        X_num=[[1]],
        X_nom=[[]],
        y_codes=[],
        y=[],
    )

    with pytest.raises(ValueError, match="training data must not be empty"):
        knn_algorithm(
            train=train,
            test=test,
            k=1,
            distance_function=lambda a, b: 0.0,
            voting_function=first_neighbor_label,
        )


def test_knn_algorithm_rejects_k_larger_than_training_size():
    train = FakeDataset(
        X_num=[[1], [2]],
        X_nom=[[], []],
        y_codes=[0, 1],
        y=[0, 1],
    )
    test = FakeDataset(
        X_num=[[1]],
        X_nom=[[]],
        y_codes=[],
        y=[],
    )

    with pytest.raises(ValueError, match="k cannot exceed"):
        knn_algorithm(
            train=train,
            test=test,
            k=3,
            distance_function=lambda a, b: abs(a[0][0] - b[0][0]),
            voting_function=first_neighbor_label,
        )


# ---------------------------------------------------------------------------
# Experiment adapter
# ---------------------------------------------------------------------------


def test_make_knn_classifier_returns_experiment_result():
    train = FakeDataset(
        X_num=[[1], [5], [9]],
        X_nom=[[], [], []],
        y_codes=[0, 1, 1],
        y=[0, 1, 1],
    )
    test = FakeDataset(
        X_num=[[2], [8]],
        X_nom=[[], []],
        y_codes=[],
        y=[],
    )

    classifier = make_knn_classifier(
        k=1,
        distance_function=lambda a, b: abs(a[0][0] - b[0][0]),
        voting_function=first_neighbor_label,
    )

    result = classifier(train, test)

    assert result.predictions == [0, 1]
    assert result.storage == len(train.y)


def test_make_knn_classifier_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be a positive integer"):
        make_knn_classifier(
            k=0,
            distance_function=absolute_distance,
            voting_function=first_neighbor_label,
        )
