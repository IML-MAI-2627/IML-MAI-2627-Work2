"""Core utilities for kNN: data containers, distance matrices, retrieval, and execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from models.knn.retention import OblivionByGoodness, RetentionFunction
from preprocessing import Preprocessed

# ---------------------------------------------------------------------------
# Dataclasses & Type Aliases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Neighbor:
    """One training instance selected as a neighbor of a query instance.

    Used by: voting schemes (majority, inverse_distance), nearest neighbor sorting,
    and reduction consistency checks.
    """

    index: int
    distance: float
    label: int


@dataclass(frozen=True)
class Neighbors:
    """The k_max nearest training cases of every test case, ordered by (distance, index).

    Used by: nearest_neighbors() and predict() in the fast batch kNN path.
    """

    indices: np.ndarray  # (n_test, k_max) training-row indices
    distances: np.ndarray  # (n_test, k_max)
    labels: np.ndarray  # (n_test, k_max) training class codes


@dataclass(frozen=True)
class KNNRun:
    """Predictions plus the number of cases stored at the end of the test phase.

    Used by: run_kNN() and KNN.fit_predict() to package output and storage size.
    """

    predictions: list[int]
    case_base_size: int
    scores: np.ndarray | None = None  # (n_test, n_classes) vote shares, only if requested


DistanceFunction = Callable[..., np.ndarray]
VotingFunction = Callable[[Sequence[Neighbor]], int]
ScoreFunction = Callable[[Sequence[Neighbor], int], np.ndarray]


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------


def check_k(k: object, limit: int | None = None) -> int:
    """Validate neighborhood size k as a strictly positive integer.

    Used by: KNN.parse_params(), run_kNN(), and all reduction methods (FCNN, ENNTH, DROP).
    """
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k <= 0:
        raise ValueError("k must be a positive integer")
    if limit is not None and k > limit:
        raise ValueError(f"k cannot exceed {limit}")
    return int(k)


def _check_distances(distances: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """Verify that the computed distance matrix has expected shape, is finite and non-negative.

    Used by: nearest_neighbors(), compute_distance_matrix(), and retention loops.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.shape != expected_shape:
        raise ValueError(
            f"distance_function must return shape {expected_shape}, got {distances.shape}"
        )
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distance_function must return finite, non-negative distances")
    return distances


def _check_vote(prediction: object) -> int:
    """Ensure the voting function returns a valid integer class code.

    Used by: predict(), classify_from_neighbors(), and _run_with_retention().
    """
    if isinstance(prediction, bool) or not isinstance(prediction, int | np.integer):
        raise ValueError("voting_function must return an integer class code")
    return int(prediction)


# ---------------------------------------------------------------------------
# Preprocessed Data Slicing & Concatenation
# ---------------------------------------------------------------------------


def concat_preprocessed(a: Preprocessed, b: Preprocessed) -> Preprocessed:
    """Concatenate two Preprocessed datasets along the instance axis.

    Used by: compute_full_distance_matrix() to compute pairwise distances across folds.
    """
    return Preprocessed(
        X_num=np.vstack([a.X_num, b.X_num]),
        X_nom=np.vstack([a.X_nom, b.X_nom]),
        y=np.concatenate([a.y, b.y]),
        y_codes=np.concatenate([a.y_codes, b.y_codes]),
        num_names=a.num_names,
        nom_names=a.nom_names,
        nom_values=a.nom_values,
        class_values=a.class_values,
    )


def subset_preprocessed(data: Preprocessed, indices: np.ndarray | Sequence[int]) -> Preprocessed:
    """Extract a subset of rows from a Preprocessed object given their indices.

    Used by: reduction algorithms (ReductionResult.data) and KNN.fit_predict().
    """
    idx = np.asarray(indices, dtype=np.int64)
    return Preprocessed(
        X_num=data.X_num[idx],
        X_nom=data.X_nom[idx],
        y=data.y[idx],
        y_codes=data.y_codes[idx],
        num_names=data.num_names,
        nom_names=data.nom_names,
        nom_values=data.nom_values,
        class_values=data.class_values,
    )


# ---------------------------------------------------------------------------
# Pairwise Distance Matrix Computations
# ---------------------------------------------------------------------------


def compute_distance_matrix(data: Preprocessed, distance_fn: DistanceFunction) -> np.ndarray:
    """Calculate the symmetric (n, n) pairwise distance matrix with zero diagonal.

    Used by: FCNN, ENNTH, and DROP to evaluate neighborhood relationships.
    """
    n = len(data.y_codes)
    try:
        D = distance_fn(data)
    except TypeError:
        D = distance_fn(data, data)
    D = _check_distances(D, (n, n))

    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


def compute_full_distance_matrix(
    train: Preprocessed,
    test: Preprocessed,
    distance_fn: DistanceFunction,
) -> np.ndarray:
    """Compute the combined (n_train + n_test, n_train + n_test) symmetric distance matrix once.

    Used by: KNN.fit_predict() when reduction is requested.
    """
    combined = concat_preprocessed(train, test)
    return compute_distance_matrix(combined, distance_fn)


# ---------------------------------------------------------------------------
# Neighbor Retrieval & Classification Helpers
# ---------------------------------------------------------------------------


def classify_from_neighbors(
    query_idx: int,
    neighbor_indices: Sequence[int],
    D: np.ndarray,
    labels: np.ndarray,
    voting_fn: VotingFunction,
) -> int:
    """Classify a query instance using a given list of neighbor indices and a voting function.

    Used by: FCNN (consistency checking), ENNTH (tie breaking), and DROP (leave-one-out checking).
    """
    nb_objects = [
        Neighbor(
            index=int(j),
            distance=float(D[query_idx, j]),
            label=int(labels[j]),
        )
        for j in neighbor_indices
    ]
    return _check_vote(voting_fn(nb_objects))


def find_k_nearest(
    D: np.ndarray,
    query_idx: int,
    candidate_indices: Sequence[int],
    k: int,
    exclude_self: bool = False,
) -> list[int]:
    """Find the k nearest candidate instances to query_idx based on distance matrix D.

    Used by: reduction algorithms to query prototypes from subsets of candidates.
    """
    candidates = [c for c in candidate_indices if not (exclude_self and c == query_idx)]
    if not candidates:
        return []
    k_actual = min(check_k(k), len(candidates))
    dists = [D[query_idx, c] for c in candidates]
    order = np.argsort(dists, kind="stable")[:k_actual]
    return [candidates[i] for i in order]


def nearest_neighbors(
    train: Preprocessed,
    test: Preprocessed,
    distance_function: DistanceFunction,
    k_max: int,
    distance_matrix: np.ndarray | None = None,
) -> Neighbors:
    """Compute or receive distances and extract the k_max nearest training cases per test case.

    Used by: run_kNN() in the fast batch path without retention.
    """
    n_train = len(train.y_codes)
    if n_train == 0:
        raise ValueError("training data must not be empty")
    k_max = check_k(k_max, n_train)
    n_test = len(test.y_codes)

    if distance_matrix is not None:
        distances = _check_distances(distance_matrix, (n_test, n_train))
    else:
        distances = _check_distances(distance_function(train, test), (n_test, n_train))

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
    score_function: ScoreFunction | None = None,
    n_classes: int = 0,
) -> list[int] | tuple[list[int], np.ndarray]:
    """Predict one class code per test case using its first k nearest neighbors.

    With ``score_function`` it also returns the (n_test, n_classes) vote shares.
    Used by: run_kNN() in the fast batch path without retention.
    """
    k = check_k(k, neighbors.indices.shape[1])
    predictions = []
    scores = []

    for indices, distances, labels in zip(
        neighbors.indices, neighbors.distances, neighbors.labels, strict=True
    ):
        nearest = [
            Neighbor(index=int(i), distance=float(d), label=int(c))
            for i, d, c in zip(indices[:k], distances[:k], labels[:k], strict=True)
        ]
        predictions.append(_check_vote(voting_function(nearest)))
        if score_function is not None:
            scores.append(score_function(nearest, n_classes))

    if score_function is not None:
        return predictions, np.asarray(scores)
    return predictions


# ---------------------------------------------------------------------------
# kNN Execution (Batch & Dynamic Retention)
# ---------------------------------------------------------------------------


def _run_with_retention(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction | OblivionByGoodness,
    distance_matrix: np.ndarray | None = None,
    score_function: ScoreFunction | None = None,
    n_classes: int = 0,
) -> KNNRun:
    """Solve test cases sequentially, updating the case base according to retention policy.

    Used by: run_kNN() and KNN.fit_predict() when retention is enabled.
    """
    n_train, n_test = len(train.y_codes), len(test.y_codes)
    if n_train == 0:
        raise ValueError("training data must not be empty")
    k = check_k(k, n_train)

    if distance_matrix is not None:
        base_distances = _check_distances(distance_matrix, (n_test, n_train + n_test))
    else:
        to_train = _check_distances(distance_function(train, test), (n_test, n_train))
        try:
            to_test = distance_function(test)
        except TypeError:
            to_test = distance_function(test, test)
        to_test = _check_distances(to_test, (n_test, n_test))
        base_distances = np.hstack([to_train, to_test])

    base_labels = np.concatenate([train.y_codes, test.y_codes])
    active = np.zeros(n_train + n_test, dtype=bool)
    active[:n_train] = True

    og = retention_function if isinstance(retention_function, OblivionByGoodness) else None
    goodness = og.init_goodness(n_train + n_test) if og is not None else np.zeros(n_train + n_test)

    predictions = []
    scores = []
    for i in range(n_test):
        candidates = np.flatnonzero(active)
        order = candidates[np.argsort(base_distances[i, candidates], kind="stable")[:k]]
        nearest = [
            Neighbor(index=int(j), distance=float(base_distances[i, j]), label=int(base_labels[j]))
            for j in order
        ]
        prediction = _check_vote(voting_function(nearest))
        predictions.append(prediction)
        if score_function is not None:
            scores.append(score_function(nearest, n_classes))
        true_label = int(test.y_codes[i])
        if og is not None:
            for j in order:
                og.update_neighbor(j, base_labels[j] == true_label, goodness, active, min_cases=k)
            active[n_train + i] = True
        elif retention_function(prediction, true_label):
            active[n_train + i] = True

    return KNNRun(
        predictions=predictions,
        case_base_size=int(active.sum()),
        scores=np.asarray(scores) if score_function is not None else None,
    )


def run_kNN(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction | OblivionByGoodness | None = None,
    distance_matrix: np.ndarray | None = None,
    score_function: ScoreFunction | None = None,
) -> KNNRun:
    """Run kNN classification with optional retention and optional precomputed distances.

    ``score_function`` (optional) also returns per-class vote shares, for ROC / PR curves.
    Used by: KNN.fit_predict() to execute standard or precomputed kNN evaluation.
    """
    n_classes = len(getattr(train, "class_values", ())) or (
        int(np.max(train.y_codes)) + 1 if len(train.y_codes) > 0 else 0
    )
    if retention_function is None:
        neighbors = nearest_neighbors(
            train, test, distance_function, k_max=k, distance_matrix=distance_matrix
        )
        if score_function is None:
            return KNNRun(predict(neighbors, k, voting_function), len(train.y_codes))
        predictions, scores = predict(neighbors, k, voting_function, score_function, n_classes)
        return KNNRun(predictions, len(train.y_codes), scores)
    return _run_with_retention(
        train,
        test,
        k,
        distance_function,
        voting_function,
        retention_function,
        distance_matrix=distance_matrix,
        score_function=score_function,
        n_classes=n_classes,
    )


def kNNAlgorithm(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: DistanceFunction,
    voting_function: VotingFunction,
    retention_function: RetentionFunction | OblivionByGoodness | None = None,
) -> list[int]:
    """Predict one class code per test case with the k nearest cases of the case base.

    Named as the assignment requires (``kNNAlgorithm(...)``). Without a retention policy the
    case base is the training set; with one, the test cases are solved in order.
    """
    return run_kNN(
        train, test, k, distance_function, voting_function, retention_function
    ).predictions
