"""Basic, library-independent building blocks for k-nearest neighbours."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import nsmallest


@dataclass(frozen=True)
class Neighbor:
    """One training instance selected as a neighbor of a query instance."""

    index: int
    distance: float
    label: int


DistanceFunction = Callable[[object, object], float]
VotingFunction = Callable[[Sequence[Neighbor]], int]


def find_neighbors(
    train_X: Sequence[object],
    train_y: Sequence[int],
    query: object,
    k: int,
    distance_function: DistanceFunction,
) -> list[Neighbor]:
    """Return the k closest training instances, ordered by distance then index."""

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if len(train_X) == 0:
        raise ValueError("training data must not be empty")

    if len(train_X) != len(train_y):
        raise ValueError("train_X and train_y must contain the same number of instances")

    if k > len(train_X):
        raise ValueError("k cannot exceed the number of training instances")

    def candidates():
        """Yield validated candidates in original training order."""
        for index, (instance, label) in enumerate(zip(train_X, train_y, strict=True)):
            distance = float(distance_function(query, instance))

            if not math.isfinite(distance) or distance < 0:
                raise ValueError("distance_function must return finite, non-negative distances")

            yield Neighbor(index=index, distance=distance, label=int(label))

    # ``nsmallest`` keeps only k candidates in its heap, avoiding a full O(n log n)
    # sort for every test case. The index supplies a stable, deterministic tie-break.
    return nsmallest(k, candidates(), key=lambda neighbor: (neighbor.distance, neighbor.index))


def knn_algorithm(
    train: object,
    test: object,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
) -> list[int]:
    """Predict class codes using a configurable kNN pipeline."""

    train_X = [
        (num, nom)
        for num, nom in zip(
            train.X_num,
            train.X_nom,
            strict=True,
        )
    ]
    train_y = train.y_codes

    test_X = [
        (num, nom)
        for num, nom in zip(
            test.X_num,
            test.X_nom,
            strict=True,
        )
    ]

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if len(train_X) == 0:
        raise ValueError("training data must not be empty")

    if len(train_X) != len(train_y):
        raise ValueError("training features and labels must have the same number of instances")

    if k > len(train_X):
        raise ValueError("k cannot exceed the number of training instances")

    predictions = []

    for query in test_X:
        neighbors = find_neighbors(
            train_X=train_X,
            train_y=train_y,
            query=query,
            k=k,
            distance_function=distance_function,
        )

        prediction = voting_function(neighbors)
        predictions.append(int(prediction))

    return predictions


def make_knn_classifier(
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
) -> Callable[[object, object], object]:
    """Create a kNN classifier compatible with the experiment runner."""

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    def classifier(train: object, test: object) -> object:
        from experiment_runner import ClassificationResult

        predictions = knn_algorithm(
            train=train,
            test=test,
            k=k,
            distance_function=distance_function,
            voting_function=voting_function,
        )

        return ClassificationResult(
            predictions=predictions,
            storage=len(train.y),
        )

    return classifier
