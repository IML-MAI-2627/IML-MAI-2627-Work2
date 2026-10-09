"""From-scratch k-nearest neighbours (K-IBL): neighbours computed once per distance and fold."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from experiment_runner import ClassificationResult, Classifier
from preprocessing import Preprocessed


@dataclass(frozen=True)
class Neighbor:
    """One training instance selected as a neighbor of a query instance."""

    index: int
    distance: float
    label: int


@dataclass(frozen=True)
class Neighbors:
    """The k_max nearest training cases of every test case, ordered by (distance, index)."""

    indices: np.ndarray  # (n_test, k_max) training-row indices
    distances: np.ndarray  # (n_test, k_max)
    labels: np.ndarray  # (n_test, k_max) training class codes


DistanceFunction = Callable[[Preprocessed, Preprocessed], np.ndarray]
VotingFunction = Callable[[Sequence[Neighbor]], int]


def check_k(k: object, limit: int | None = None) -> int:
    """Validate k (Python or numpy integer) and return it as a Python int."""
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k <= 0:
        raise ValueError("k must be a positive integer")
    if limit is not None and k > limit:
        raise ValueError(f"k cannot exceed {limit}")
    return int(k)


def nearest_neighbors(
    train: Preprocessed,
    test: Preprocessed,
    distance_function: DistanceFunction,
    k_max: int,
) -> Neighbors:
    """Compute all distances once and keep the k_max nearest training cases per test case."""
    n_train = len(train.y_codes)
    if n_train == 0:
        raise ValueError("training data must not be empty")
    k_max = check_k(k_max, n_train)

    distances = np.asarray(distance_function(train, test), dtype=float)
    expected_shape = (len(test.y_codes), n_train)
    if distances.shape != expected_shape:
        raise ValueError(
            f"distance_function must return shape {expected_shape}, got {distances.shape}"
        )
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distance_function must return finite, non-negative distances")

    # A stable sort keeps the lower training index first when distances are equal.
    order = np.argsort(distances, axis=1, kind="stable")[:, :k_max]
    return Neighbors(
        indices=order,
        distances=np.take_along_axis(distances, order, axis=1),
        labels=np.asarray(train.y_codes)[order],
    )


def predict(
    neighbors: Neighbors,
    k: int,
    voting_function: VotingFunction,
) -> list[int]:
    """Predict one class code per test case using its first k neighbours."""
    k = check_k(k, neighbors.indices.shape[1])
    predictions = []

    for indices, distances, labels in zip(
        neighbors.indices, neighbors.distances, neighbors.labels, strict=True
    ):
        nearest = [
            Neighbor(index=int(i), distance=float(d), label=int(c))
            for i, d, c in zip(indices[:k], distances[:k], labels[:k], strict=True)
        ]
        prediction = voting_function(nearest)
        if isinstance(prediction, bool) or not isinstance(prediction, int | np.integer):
            raise ValueError("voting_function must return an integer class code")
        predictions.append(int(prediction))

    return predictions


def knn_algorithm(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
) -> list[int]:
    """Predict one class code per test case (same entry point as knn_base)."""
    neighbors = nearest_neighbors(train, test, distance_function, k_max=k)
    return predict(neighbors, k, voting_function)


def make_knn_classifier(
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
) -> Classifier:
    """Create a kNN classifier compatible with the experiment runner."""
    k = check_k(k)

    def classifier(train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        predictions = knn_algorithm(train, test, k, distance_function, voting_function)
        return ClassificationResult(predictions=predictions, storage=len(train.y))

    return classifier
