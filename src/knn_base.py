"""From-scratch k-nearest neighbours (K-IBL) core: neighbour retrieval and ``kNNAlgorithm``.

Distances are computed once per fold as an (n_test, n_train) matrix with numpy, so the
neighbours of every test case can be reused for every K. Distance metrics and voting
policies are injected callables and live in their own modules (``distances``, ``voting``).
The runner-facing adapter is ``models.knn.KNN``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

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
RetentionFunction = Callable[[int, int], bool]


def check_k(k: object, limit: int | None = None) -> int:
    """Validate k (Python or numpy integer) and return it as a Python int."""
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k <= 0:
        raise ValueError("k must be a positive integer")
    if limit is not None and k > limit:
        raise ValueError(f"k cannot exceed {limit}")
    return int(k)


def _check_distances(distances: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """Return the distance matrix as floats, or fail if its shape or values are wrong."""
    distances = np.asarray(distances, dtype=float)
    if distances.shape != expected_shape:
        raise ValueError(
            f"distance_function must return shape {expected_shape}, got {distances.shape}"
        )
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distance_function must return finite, non-negative distances")
    return distances


def _check_vote(prediction: object) -> int:
    """Return the vote as a Python int, or fail if it is not an integer class code."""
    if isinstance(prediction, bool) or not isinstance(prediction, int | np.integer):
        raise ValueError("voting_function must return an integer class code")
    return int(prediction)


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
    distances = _check_distances(distance_function(train, test), (len(test.y_codes), n_train))

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
        predictions.append(_check_vote(voting_function(nearest)))

    return predictions


@dataclass(frozen=True)
class KNNRun:
    """Predictions plus the number of cases stored at the end of the test phase."""

    predictions: list[int]
    case_base_size: int


def _run_with_retention(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction,
) -> KNNRun:
    """Solve the test cases one by one, in order, and retain them as the policy says.

    The case base is [train ; test]: row j < n_train is training case j and row n_train + i
    is test case i, retained with its true class. Test case i can only see the test cases
    retained before it. All distances are computed once; ``active`` marks the cases stored.
    """
    n_train, n_test = len(train.y_codes), len(test.y_codes)
    if n_train == 0:
        raise ValueError("training data must not be empty")
    k = check_k(k, n_train)

    to_train = _check_distances(distance_function(train, test), (n_test, n_train))
    to_test = _check_distances(distance_function(test, test), (n_test, n_test))
    base_distances = np.hstack([to_train, to_test])  # (n_test, n_train + n_test)
    base_labels = np.concatenate([train.y_codes, test.y_codes])
    active = np.zeros(n_train + n_test, dtype=bool)
    active[:n_train] = True

    predictions = []
    for i in range(n_test):
        candidates = np.flatnonzero(active)  # increasing index
        # Stable sort over increasing indices: equal distances keep the lower index first.
        order = candidates[np.argsort(base_distances[i, candidates], kind="stable")[:k]]
        nearest = [
            Neighbor(index=int(j), distance=float(base_distances[i, j]), label=int(base_labels[j]))
            for j in order
        ]
        prediction = _check_vote(voting_function(nearest))
        predictions.append(prediction)
        if retention_function(prediction, int(test.y_codes[i])):
            active[n_train + i] = True

    return KNNRun(predictions=predictions, case_base_size=int(active.sum()))


def run_kNN(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction | None = None,
) -> KNNRun:
    """kNN with an optional retention policy; None means never retain (fast batch path)."""
    if retention_function is None:
        neighbors = nearest_neighbors(train, test, distance_function, k_max=k)
        return KNNRun(predict(neighbors, k, voting_function), len(train.y_codes))
    return _run_with_retention(
        train, test, k, distance_function, voting_function, retention_function
    )


def kNNAlgorithm(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction | None = None,
) -> list[int]:
    """Predict one class code per test case with the k nearest cases of the case base.

    Named as the assignment requires (``kNNAlgorithm(...)``). Without a retention policy the
    case base is the training set; with one, the test cases are solved in order.
    """
    return run_kNN(
        train, test, k, distance_function, voting_function, retention_function
    ).predictions
