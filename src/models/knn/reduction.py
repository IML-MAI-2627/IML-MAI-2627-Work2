"""Instance reduction techniques for nearest neighbor classifiers using only NumPy.

This module implements the training-set reduction algorithms from the literature:
1. Fast Condensed Nearest Neighbor (FCNN: FCNN1, FCNN2, FCNN3, FCNN4, and k-FCNN)
   Reference: Angiulli, F. (2007). "Fast Nearest Neighbor Condensation for Large
   Data Sets Classification", IEEE TKDE 19(11): 1450-1464.
2. Extended Edited Nearest Neighbor with Threshold (EENTh / WilsonTh and WilsonProb)
   Reference: Vázquez, F., Sánchez, J.S., Pla, F. (2005). "A Stochastic Approach to
   Wilson's Editing Algorithm", IbPRIA 2005, LNCS 3523: 35-42.
3. Decremental Reduction Optimization Procedure (DROP: DROP1 through DROP5, with DROP3
   as the primary advanced rule)
   Reference: Wilson, D.R., Martinez, T.R. (2000). "Reduction Techniques for
   Instance-Based Learning Algorithms", Machine Learning 38(3): 257-286.

All methods are modular: users can configure the distance metric (by name or callable),
the voting scheme (by name or callable), and neighborhood size k. The algorithms work
seamlessly with the project's ``Preprocessed`` folds as well as raw NumPy arrays (X, y).
Only NumPy is used for all reduction routines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from models.knn.distances import DISTANCES
from models.knn.knn_base import Neighbor, check_k
from models.knn.voting import VOTING
from preprocessing import Preprocessed

# ---------------------------------------------------------------------------
# Data Structures and Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReductionResult:
    """Outcome of an instance reduction algorithm.

    Attributes:
        indices: 1D NumPy array of sorted indices retained from the original dataset.
        mask: 1D boolean NumPy array where True indicates retained instances.
        n_original: Total instances before reduction.
        n_retained: Total instances retained after reduction.
        reduction_rate: Fraction of removed instances (1.0 - n_retained / n_original).
        data: The reduced dataset in the same representation as the input
              (a filtered ``Preprocessed`` object or a tuple ``(X_reduced, y_reduced)``).
    """

    indices: np.ndarray
    mask: np.ndarray
    n_original: int
    n_retained: int
    reduction_rate: float
    data: Preprocessed | tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Input Normalization and Slicing Helpers
# ---------------------------------------------------------------------------


def _subset_preprocessed(data: Preprocessed, indices: np.ndarray) -> Preprocessed:
    """Return a new Preprocessed object containing only the specified row indices."""
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


def _array_to_preprocessed(X: np.ndarray, y: np.ndarray) -> Preprocessed:
    """Convert raw NumPy (X, y) into a Preprocessed representation for unified handling."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    unique_labels, y_codes = np.unique(y, return_inverse=True)
    class_values = tuple(str(lbl) for lbl in unique_labels)
    return Preprocessed(
        X_num=X,
        X_nom=np.empty((len(X), 0), dtype=np.int64),
        y=np.asarray([str(lbl) for lbl in y]),
        y_codes=y_codes.astype(np.int64),
        num_names=tuple(f"f{i}" for i in range(X.shape[1])),
        nom_names=(),
        nom_values=(),
        class_values=class_values,
    )


def _unpack_input(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
) -> tuple[Preprocessed, bool]:
    """Normalize input into (Preprocessed, is_raw_tuple)."""
    if isinstance(data, Preprocessed):
        return data, False
    if y is not None:
        return _array_to_preprocessed(data, y), True
    if isinstance(data, tuple) and len(data) == 2:
        return _array_to_preprocessed(data[0], data[1]), True
    raise TypeError(
        "Expected data to be a Preprocessed instance, a (X, y) tuple, or X with y provided"
    )


def _wrap_result(
    prep: Preprocessed,
    indices: np.ndarray,
    is_raw: bool,
) -> ReductionResult:
    """Build the ReductionResult container."""
    n_orig = len(prep.y_codes)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    mask = np.zeros(n_orig, dtype=bool)
    mask[indices] = True
    n_ret = len(indices)
    rate = 1.0 - (n_ret / n_orig) if n_orig > 0 else 0.0

    if is_raw:
        reduced_data = (prep.X_num[indices], prep.y_codes[indices])
    else:
        reduced_data = _subset_preprocessed(prep, indices)

    return ReductionResult(
        indices=indices,
        mask=mask,
        n_original=n_orig,
        n_retained=n_ret,
        reduction_rate=rate,
        data=reduced_data,
    )


# ---------------------------------------------------------------------------
# Distance and Voting Resolvers
# ---------------------------------------------------------------------------


def resolve_distance(
    distance: str | Callable[..., np.ndarray],
) -> Callable[[Preprocessed, Preprocessed], np.ndarray]:
    """Resolve distance metric from name or callable."""
    if isinstance(distance, str):
        key = distance.lower()
        if key not in DISTANCES:
            raise ValueError(f"Unknown distance '{distance}'. Choose from {sorted(DISTANCES)}.")
        return DISTANCES[key]
    if callable(distance):
        return distance
    raise TypeError(f"distance must be a string or callable, got {type(distance).__name__}")


def resolve_voting(
    voting: str | Callable[[Sequence[Neighbor]], int],
) -> Callable[[Sequence[Neighbor]], int]:
    """Resolve voting policy from name or callable."""
    if isinstance(voting, str):
        key = voting.lower()
        if key not in VOTING:
            raise ValueError(f"Unknown voting '{voting}'. Choose from {sorted(VOTING)}.")
        return VOTING[key]
    if callable(voting):
        return voting
    raise TypeError(f"voting must be a string or callable, got {type(voting).__name__}")


def compute_distance_matrix(
    data: Preprocessed,
    distance_fn: Callable[[Preprocessed, Preprocessed], np.ndarray],
) -> np.ndarray:
    """Compute (N, N) distance matrix using the distance function and NumPy."""
    n = len(data.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    # In distances.py, signature is distance_fn(train, test) -> (n_test, n_train).
    # When passed (data, data), it returns (N, N) pairwise distances.
    D = np.asarray(distance_fn(data, data), dtype=float)
    if D.shape != (n, n):
        raise ValueError(f"Expected distance matrix of shape ({n}, {n}), got {D.shape}")
    # Numerical safety: diagonal is 0, clip negative rounding artifacts
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def _predict_neighbors(
    query_idx: int,
    neighbor_indices: Sequence[int],
    D: np.ndarray,
    labels: np.ndarray,
    voting_fn: Callable[[Sequence[Neighbor]], int],
) -> int:
    """Predict label of a query given a sequence of neighbor indices."""
    nb_list = [
        Neighbor(
            index=int(j),
            distance=float(D[query_idx, j]),
            label=int(labels[j]),
        )
        for j in neighbor_indices
    ]
    return int(voting_fn(nb_list))


# ---------------------------------------------------------------------------
# 1. Fast Condensed Nearest Neighbor (FCNN)
# ---------------------------------------------------------------------------


def fcnn(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    k: int = 1,
    variant: Literal["fcnn1", "fcnn2", "fcnn3", "fcnn4"] = "fcnn1",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Fast Condensed Nearest Neighbor (FCNN) rule (Angiulli, 2007).

    Computes a training-set-consistent subset for nearest neighbor classification.
    Starts with the class centroids and iteratively adds representatives of the
    Voronoi enemies in the current subset until consistency is achieved.

    Args:
        data: Preprocessed dataset or (X, y) tuple.
        y: Optional labels array if data is feature matrix X.
        distance: Distance metric name in DISTANCES or callable.
        k: Neighborhood size for k-training-set consistency (default 1).
        variant: FCNN algorithm variant:
            - 'fcnn1': Representative is the nearest Voronoi enemy (default).
            - 'fcnn2': Representative is the closest class centroid of Voronoi enemies.
            - 'fcnn3': Adds 1 instance per iteration (nearest enemy of prototype with max enemies).
            - 'fcnn4': Adds 1 instance per iteration (centroid enemy of prototype with max enemies).
        distance_matrix: Optional precomputed (N, N) pairwise distance matrix.

    Returns:
        ReductionResult with retained indices and reduced dataset.
    """
    prep, is_raw = _unpack_input(data, y)
    n = len(prep.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    k = check_k(k, n)

    variant = variant.lower()
    if variant not in ("fcnn1", "fcnn2", "fcnn3", "fcnn4"):
        raise ValueError(
            f"Unknown variant '{variant}'. Expected 'fcnn1', 'fcnn2', 'fcnn3', or 'fcnn4'."
        )

    dist_fn = resolve_distance(distance)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(prep, dist_fn)
    labels = prep.y_codes
    classes = np.unique(labels)

    # Helper: compute class centroid in a subset of indices.
    # In Angiulli (2007), centroid is the point closest to the geometric center,
    # which in metric distance corresponds to minimizing sum of squared distances.
    def get_class_centroid(subset_indices: np.ndarray | list[int]) -> int:
        sub = np.asarray(subset_indices, dtype=np.int64)
        if len(sub) == 1:
            return int(sub[0])
        sub_D = D[np.ix_(sub, sub)]
        scores = np.sum(sub_D**2, axis=1)
        # Lexicographical / stable tie-break by lowest index
        best_local = np.argmin(scores)
        return int(sub[best_local])

    # Initial seeds:
    # FCNN1 & FCNN2: centroids of all classes.
    # FCNN3 & FCNN4: centroid of the most populated class.
    if variant in ("fcnn3", "fcnn4"):
        class_counts = {c: int(np.sum(labels == c)) for c in classes}
        most_pop_class = max(class_counts, key=class_counts.get)
        initial_seeds = [get_class_centroid(np.where(labels == most_pop_class)[0])]
    else:
        initial_seeds = [get_class_centroid(np.where(labels == c)[0]) for c in classes]

    S: set[int] = set(initial_seeds)
    delta_S: list[int] = sorted(initial_seeds)

    nearest = np.full(n, -1, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float64)

    # For points in initial S, nearest is self
    for p in S:
        nearest[p] = p
        min_dist[p] = 0.0

    while len(delta_S) > 0:
        S.update(delta_S)
        rem = np.array([i for i in range(n) if i not in S], dtype=np.int64)
        if len(rem) == 0:
            break

        # Vectorized nearest update: compare remaining points with new delta_S
        delta_arr = np.array(delta_S, dtype=np.int64)
        delta_dists = D[np.ix_(rem, delta_arr)]  # (len(rem), len(delta_S))
        best_delta_col = np.argmin(delta_dists, axis=1)
        best_delta_dist = delta_dists[np.arange(len(rem)), best_delta_col]
        best_delta_p = delta_arr[best_delta_col]

        closer_mask = best_delta_dist < min_dist[rem]
        closer_rem = rem[closer_mask]
        nearest[closer_rem] = best_delta_p[closer_mask]
        min_dist[closer_rem] = best_delta_dist[closer_mask]

        # Voronoi enemies: points q in rem whose k-NN (or 1-NN) in S misclassifies q.
        # q belongs to the Voronoi cell of nearest[q].
        voren: dict[int, list[int]] = {p: [] for p in S}

        if k == 1:
            for q in rem:
                p_star = nearest[q]
                if labels[q] != labels[p_star]:
                    voren[p_star].append(q)
        else:
            # k-FCNN consistency: check k-NN prediction under S
            S_list = np.array(sorted(S), dtype=np.int64)
            k_eff = min(k, len(S_list))
            sub_D = D[np.ix_(rem, S_list)]
            nb_order = np.argpartition(sub_D, k_eff - 1, axis=1)[:, :k_eff]
            for row_idx, q in enumerate(rem):
                nb_indices = S_list[nb_order[row_idx]]
                # majority vote
                counts = Counter(labels[nb_indices])
                pred_label = max(counts, key=counts.get)
                if labels[q] != pred_label:
                    p_star = nearest[q]
                    voren[p_star].append(q)

        # Select representatives
        if variant == "fcnn1":
            rep: dict[int, int] = {}
            for p, enemies in voren.items():
                if enemies:
                    # Closest Voronoi enemy to p
                    best_enemy = enemies[np.argmin(D[p, enemies])]
                    rep[p] = best_enemy
            delta_S = sorted(set(rep.values()))

        elif variant == "fcnn2":
            rep = {}
            for p, enemies in voren.items():
                if enemies:
                    e_arr = np.array(enemies, dtype=np.int64)
                    e_classes = np.unique(labels[e_arr])
                    centroids = [get_class_centroid(e_arr[labels[e_arr] == c]) for c in e_classes]
                    # Select centroid closest to p
                    best_c = centroids[np.argmin(D[p, centroids])]
                    rep[p] = best_c
            delta_S = sorted(set(rep.values()))

        elif variant in ("fcnn3", "fcnn4"):
            # Select only prototype p_star with maximum number of Voronoi enemies
            max_enemies = -1
            best_p = None
            for p in sorted(S):
                num_e = len(voren[p])
                if num_e > max_enemies:
                    max_enemies = num_e
                    best_p = p

            if best_p is not None and max_enemies > 0:
                enemies = voren[best_p]
                if variant == "fcnn3":
                    chosen = enemies[np.argmin(D[best_p, enemies])]
                else:
                    e_arr = np.array(enemies, dtype=np.int64)
                    e_classes = np.unique(labels[e_arr])
                    centroids = [get_class_centroid(e_arr[labels[e_arr] == c]) for c in e_classes]
                    chosen = centroids[np.argmin(D[best_p, centroids])]
                delta_S = [chosen]
            else:
                delta_S = []

    return _wrap_result(prep, np.array(sorted(S), dtype=np.int64), is_raw)


# ---------------------------------------------------------------------------
# 2. Extended Edited Nearest Neighbor with Threshold (EENTh)
# ---------------------------------------------------------------------------


def eenth(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    k: int = 3,
    threshold: float | None = 0.7,
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Extended Edited Nearest Neighbor with Threshold (EENTh / WilsonTh).

    Reference: Vázquez, Sánchez, Pla (2005). "A Stochastic Approach to Wilson's
    Editing Algorithm".

    Estimates class conditional probabilities for each prototype x using its k
    nearest neighbors weighted inversely by (1 + distance):
        P_i(x) = sum_{j in N_k(x), y_j == i} 1 / (1 + d(x, x^j))
        p_i(x) = P_i(x) / sum_m P_m(x)

    An instance is eliminated if:
        1. delta_{k-prob}(x) != y[x] (misclassified by weighted probability), OR
        2. max_i p_i(x) <= threshold (uncertain classification near border).

    Setting threshold=0.0 or None computes the WilsonProb algorithm (stochastic
    Wilson editing without threshold).

    Args:
        data: Preprocessed dataset or (X, y) tuple.
        y: Optional labels array if data is feature matrix X.
        distance: Distance metric name in DISTANCES or callable.
        k: Number of neighbors in leave-one-out neighborhood (default 3).
        threshold: Minimum confidence threshold mu in [0, 1) (default 0.7).
        voting: Voting policy used for tie-breaks.
        distance_matrix: Optional precomputed (N, N) pairwise distance matrix.

    Returns:
        ReductionResult with retained indices and edited dataset.
    """
    prep, is_raw = _unpack_input(data, y)
    n = len(prep.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    k = check_k(k, max(1, n - 1))

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(prep, dist_fn)
    labels = prep.y_codes
    classes = np.unique(labels)
    n_classes = len(classes)

    # Leave-one-out: exclude self
    D_loo = D.copy()
    np.fill_diagonal(D_loo, np.inf)

    # Top k nearest neighbors for each prototype
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return _wrap_result(prep, np.arange(n), is_raw)

    order = np.argsort(D_loo, axis=1, kind="stable")[:, :k_eff]
    nb_dists = np.take_along_axis(D_loo, order, axis=1)
    nb_labels = labels[order]

    # Weights: w_j = 1 / (1 + d(x, x^j))
    weights = 1.0 / (1.0 + nb_dists)
    total_weights = np.sum(weights, axis=1, keepdims=True)

    # Vectorized class probability accumulation
    # Map class labels to 0..n_classes-1
    label_to_idx = {c: i for i, c in enumerate(classes)}
    encoded_nb_labels = np.vectorize(label_to_idx.get)(nb_labels)

    probs = np.zeros((n, n_classes), dtype=np.float64)
    for j in range(k_eff):
        np.add.at(probs, (np.arange(n), encoded_nb_labels[:, j]), weights[:, j])

    # Normalize: p_i(x) = P_i(x) / sum_m P_m(x)
    probs = np.divide(probs, total_weights, out=probs, where=total_weights > 0)

    # Predicted class and max probability
    best_class_idx = np.argmax(probs, axis=1)
    pred_class = classes[best_class_idx]
    max_prob = np.max(probs, axis=1)

    # Deterministic tie-breaking for argmax: if multiple classes tie for max prob,
    # break tie via voting_fn on the retrieved neighbors.
    is_tied = np.sum(np.isclose(probs, max_prob[:, None]), axis=1) > 1
    if np.any(is_tied):
        for i in np.flatnonzero(is_tied):
            pred_class[i] = _predict_neighbors(i, order[i], D, labels, voting_fn)

    # Retention condition:
    # Kept if predicted class matches true label AND max_prob > threshold
    correct_mask = pred_class == labels
    if threshold is not None and threshold > 0.0:
        keep_mask = correct_mask & (max_prob > threshold)
    else:
        keep_mask = correct_mask

    retained_indices = np.flatnonzero(keep_mask)

    # Safety: do not return empty dataset
    if len(retained_indices) == 0:
        # Fallback to keeping the highest-confidence instances of each class
        fallback = []
        for c in classes:
            c_mask = np.where(labels == c)[0]
            if len(c_mask) > 0:
                fallback.append(c_mask[np.argmax(max_prob[c_mask])])
        retained_indices = np.unique(fallback)

    return _wrap_result(prep, retained_indices, is_raw)


def wilson_prob(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    k: int = 3,
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for WilsonProb (EENTh with threshold=0.0)."""
    return eenth(
        data,
        y=y,
        distance=distance,
        k=k,
        threshold=0.0,
        voting=voting,
        distance_matrix=distance_matrix,
    )


def wilson_th(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    k: int = 3,
    threshold: float = 0.7,
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for WilsonTh (EENTh with specified threshold)."""
    return eenth(
        data,
        y=y,
        distance=distance,
        k=k,
        threshold=threshold,
        voting=voting,
        distance_matrix=distance_matrix,
    )


# ---------------------------------------------------------------------------
# 3. Decremental Reduction Optimization Procedure (DROP1 - DROP5)
# ---------------------------------------------------------------------------


def _classify_instance(
    target_idx: int,
    candidate_neighbors: Sequence[int],
    D: np.ndarray,
    labels: np.ndarray,
    voting_fn: Callable[[Sequence[Neighbor]], int],
) -> int:
    """Classify target instance using given neighbor indices and voting policy."""
    nb_objects = [
        Neighbor(
            index=int(j),
            distance=float(D[target_idx, j]),
            label=int(labels[j]),
        )
        for j in candidate_neighbors
    ]
    return int(voting_fn(nb_objects))


def drop(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    version: int = 3,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Decremental Reduction Optimization Procedure (DROP1 - DROP5).

    Reference: Wilson & Martinez (2000). "Reduction Techniques for Instance-Based
    Learning Algorithms", Machine Learning 38(3): 257-286.

    Algorithms:
        - DROP1: Decremental removal, checking associates in S only.
        - DROP2: Checks associates in original training set T; sorts instances by
                 distance to nearest enemy descending (furthest first).
        - DROP3: Adds an ENN noise-filtering pass before running DROP2 (the primary,
                 most advanced rule recommended by Wilson & Martinez).
        - DROP4: Conservative noise filter (removes only if misclassified and
                 without >= with for associates), then runs DROP2.
        - DROP5: Boundary smoothing pass (nearest enemy ascending), followed by
                 repeated DROP2 pruning passes until convergence.

    Args:
        data: Preprocessed dataset or (X, y) tuple.
        y: Optional labels array if data is feature matrix X.
        version: DROP version (1, 2, 3, 4, or 5). Default 3 (most advanced).
        distance: Distance metric name in DISTANCES or callable.
        voting: Voting policy name in VOTING or callable.
        k: Number of nearest neighbors (default 3).
        distance_matrix: Optional precomputed (N, N) pairwise distance matrix.

    Returns:
        ReductionResult with retained indices and reduced dataset.
    """
    if version not in (1, 2, 3, 4, 5):
        raise ValueError(f"version must be 1, 2, 3, 4, or 5, got {version}")

    prep, is_raw = _unpack_input(data, y)
    n = len(prep.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    k = check_k(k, max(1, n - 1))

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(prep, dist_fn)
    labels = prep.y_codes

    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)

    # Initial subset S
    if version in (3, 4):
        # Noise-filtering pass using ENN with k neighbors
        k_eff = min(k, n - 1)
        enn_order = np.argsort(D_no_diag, axis=1, kind="stable")[:, :k_eff]
        misclassified: list[int] = []
        for i in range(n):
            pred = _classify_instance(i, enn_order[i], D, labels, voting_fn)
            if pred != labels[i]:
                misclassified.append(i)

        if version == 3:
            # DROP3: remove all misclassified instances
            S_set = set(i for i in range(n) if i not in misclassified)
        else:
            # DROP4: remove misclassified instance only if it does not hurt associates
            S_set = set(range(n))
            for i in misclassified:
                if len(S_set) <= k + 1:
                    break
                # If removing i does not reduce accuracy across all associates in T
                with_c = 0
                without_c = 0
                for a in range(n):
                    if a == i:
                        continue
                    # top k neighbors of a in S_set
                    a_cands = [p for p in S_set if p != a]
                    if len(a_cands) < k:
                        continue
                    a_cands_order = sorted(a_cands, key=lambda p: D[a, p])
                    top_k_with = a_cands_order[:k]
                    if i not in top_k_with:
                        continue
                    if _classify_instance(a, top_k_with, D, labels, voting_fn) == labels[a]:
                        with_c += 1
                    top_k_without = [p for p in a_cands_order if p != i][:k]
                    if _classify_instance(a, top_k_without, D, labels, voting_fn) == labels[a]:
                        without_c += 1
                if without_c >= with_c:
                    S_set.remove(i)
    else:
        S_set = set(range(n))

    # Guard: ensure S_set retains at least one instance per class if possible
    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_in_s = [p for p in S_set if labels[p] == c]
        if not c_in_s:
            # Re-insert the medoid of this class from T
            c_all = np.where(labels == c)[0]
            sub_D = D[np.ix_(c_all, c_all)]
            best = c_all[np.argmin(np.sum(sub_D**2, axis=1))]
            S_set.add(int(best))

    def compute_enemy_distances(curr_s: set[int]) -> dict[int, float]:
        """Compute distance to nearest enemy in curr_s for each prototype."""
        edist: dict[int, float] = {}
        for p in curr_s:
            enemies = [q for q in curr_s if labels[q] != labels[p]]
            edist[p] = float(np.min(D[p, enemies])) if enemies else np.inf
        return edist

    def run_removal_pass(
        curr_s: set[int],
        evaluation_order: list[int],
        track_in_s_only: bool = False,
    ) -> int:
        """Execute a decremental removal pass in the given order."""
        # Associates universe:
        # In DROP1, only instances in curr_s are tracked.
        # In DROP2-5, all instances in T are tracked.
        assoc_universe = list(curr_s) if track_in_s_only else list(range(n))

        # Build list of k+1 nearest neighbors in curr_s for each instance in assoc_universe
        neighbors: dict[int, list[int]] = {}
        for a in assoc_universe:
            cands = [p for p in curr_s if p != a]
            if len(cands) <= k:
                neighbors[a] = sorted(cands, key=lambda p: (D[a, p], p))
            else:
                cands_sorted = sorted(cands, key=lambda p: (D[a, p], p))[: k + 1]
                neighbors[a] = cands_sorted

        # Inverted index: associates[p] = {a in assoc_universe where p in neighbors[a][:k]}
        associates: dict[int, set[int]] = {p: set() for p in curr_s}
        for a in assoc_universe:
            for p in neighbors[a][:k]:
                if p in associates:
                    associates[p].add(a)

        removed_count = 0

        for p in evaluation_order:
            if p not in curr_s:
                continue
            # Never reduce below k + 1 instances total or leave any class empty
            if len(curr_s) <= k + 1:
                break
            p_class = labels[p]
            class_count_in_s = sum(1 for q in curr_s if labels[q] == p_class)
            if class_count_in_s <= 1:
                # Do not delete the last instance of a class
                continue

            p_assoc = list(associates[p])

            # Evaluate leave-one-out classification of associates: with vs without p
            with_corr = 0
            without_corr = 0
            can_remove = True

            for a in p_assoc:
                curr_nb = neighbors[a]
                if _classify_instance(a, curr_nb[:k], D, labels, voting_fn) == labels[a]:
                    with_corr += 1

                nb_without = [idx for idx in curr_nb if idx != p][:k]
                if len(nb_without) < k:
                    can_remove = False
                    break
                if _classify_instance(a, nb_without, D, labels, voting_fn) == labels[a]:
                    without_corr += 1

            if can_remove and without_corr >= with_corr:
                # Remove p from S
                curr_s.remove(p)
                removed_count += 1
                # We can remove this as we will never need it (p is already searched
                # and if it appears in others associates, we need its neighbors, not associates.
                del associates[p]

                # Update neighbor and associate lists
                for a in p_assoc:
                    neighbors[a] = [idx for idx in neighbors[a] if idx != p]
                    # Need a new neighbor so that len(neighbors[a]) == k + 1
                    curr_nb_set = set(neighbors[a])
                    new_cands = [q for q in curr_s if q != a and q not in curr_nb_set]
                    if new_cands:
                        best_new = min(new_cands, key=lambda q: (D[a, q], q))
                        neighbors[a].append(best_new)

                    # Update associate links for a's top k
                    if len(neighbors[a]) >= k:
                        kth_nb = neighbors[a][k - 1]
                        if kth_nb in associates:
                            associates[kth_nb].add(a)
                # If we have DROP1, we need to erase p as an associate
                if track_in_s_only:
                    for nbr in neighbors[p]:
                        if nbr in associates:
                            associates[nbr].discard(p)
        return removed_count

    if version == 1:
        # DROP1: original order, associates only in S
        run_removal_pass(S_set, list(S_set), track_in_s_only=True)

    elif version in (2, 3, 4):
        # DROP2 / 3 / 4: sorted by distance to nearest enemy descending (furthest first)
        edist = compute_enemy_distances(S_set)
        # Sort descending by enemy distance; break ties by index
        order = sorted(S_set, key=lambda p: (edist[p], p), reverse=True)
        run_removal_pass(S_set, order, track_in_s_only=False)

    elif version == 5:
        # DROP5:
        # Pass 1: ascending order of distance to nearest enemy (smoothing)
        edist = compute_enemy_distances(S_set)
        order = sorted(S_set, key=lambda p: (edist[p], p))
        run_removal_pass(S_set, order, track_in_s_only=False)

        # Repeated passes: descending order until convergence
        while True:
            edist = compute_enemy_distances(S_set)
            order = sorted(S_set, key=lambda p: (-edist[p], p))
            n_rem = run_removal_pass(S_set, order, track_in_s_only=False)
            if n_rem == 0:
                break

    return _wrap_result(prep, np.array(sorted(S_set), dtype=np.int64), is_raw)


def drop1(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for DROP1."""
    return drop(
        data,
        y=y,
        version=1,
        distance=distance,
        voting=voting,
        k=k,
        distance_matrix=distance_matrix,
    )


def drop2(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for DROP2."""
    return drop(
        data,
        y=y,
        version=2,
        distance=distance,
        voting=voting,
        k=k,
        distance_matrix=distance_matrix,
    )


def drop3(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for DROP3 (flagship advanced DROP algorithm)."""
    return drop(
        data,
        y=y,
        version=3,
        distance=distance,
        voting=voting,
        k=k,
        distance_matrix=distance_matrix,
    )


def drop4(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for DROP4."""
    return drop(
        data,
        y=y,
        version=4,
        distance=distance,
        voting=voting,
        k=k,
        distance_matrix=distance_matrix,
    )


def drop5(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    k: int = 3,
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Convenience alias for DROP5."""
    return drop(
        data,
        y=y,
        version=5,
        distance=distance,
        voting=voting,
        k=k,
        distance_matrix=distance_matrix,
    )


# ---------------------------------------------------------------------------
# 4. Unified Object-Oriented Interface and Registry
# ---------------------------------------------------------------------------


class InstanceReduction:
    """Unified instance reduction transformer.

    Modular interface to apply FCNN, EENTh, or DROP reduction to training data.

    Parameters:
        method: Reduction method name:
            'fcnn', 'fcnn1', 'fcnn2', 'fcnn3', 'fcnn4',
            'eenth', 'wilson_th', 'wilson_prob',
            'drop', 'drop1', 'drop2', 'drop3', 'drop4', 'drop5'.
        distance: Distance metric name or callable (default 'euclidean').
        voting: Voting policy name or callable (default 'majority').
        k: Neighborhood size (default 1 for FCNN, 3 for EENTh and DROP).
        threshold: Probability threshold for EENTh/WilsonTh (default 0.7).
        variant: Variant for FCNN ('fcnn1', 'fcnn2', 'fcnn3', 'fcnn4').
        version: Version for DROP (1..5).
    """

    def __init__(
        self,
        method: str = "drop3",
        distance: str | Callable[..., np.ndarray] = "euclidean",
        voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
        k: int | None = None,
        threshold: float | None = None,
        variant: Literal["fcnn1", "fcnn2", "fcnn3", "fcnn4"] = "fcnn1",
        version: int = 3,
    ) -> None:
        self.method = method.lower()
        self.distance = distance
        self.voting = voting
        self.k = k
        self.threshold = threshold
        self.variant = variant
        self.version = version

    def reduce(
        self,
        data: Preprocessed | tuple[np.ndarray, np.ndarray],
        y: np.ndarray | None = None,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        """Run reduction and return full ReductionResult."""
        m = self.method
        if m in ("fcnn", "fcnn1", "fcnn2", "fcnn3", "fcnn4"):
            var = m if m in ("fcnn1", "fcnn2", "fcnn3", "fcnn4") else self.variant
            k_val = self.k if self.k is not None else 1
            return fcnn(
                data,
                y=y,
                distance=self.distance,
                k=k_val,
                variant=var,
                distance_matrix=distance_matrix,
            )
        elif m in ("eenth", "wilson_th", "wilson_prob"):
            k_val = self.k if self.k is not None else 3
            th = 0.0 if m == "wilson_prob" else (0.7 if self.threshold is None else self.threshold)
            return eenth(
                data,
                y=y,
                distance=self.distance,
                k=k_val,
                threshold=th,
                voting=self.voting,
                distance_matrix=distance_matrix,
            )
        elif m in ("drop", "drop1", "drop2", "drop3", "drop4", "drop5"):
            v = int(m.replace("drop", "")) if m != "drop" else self.version
            k_val = self.k if self.k is not None else 3
            return drop(
                data,
                y=y,
                version=v,
                distance=self.distance,
                voting=self.voting,
                k=k_val,
                distance_matrix=distance_matrix,
            )
        else:
            valid = sorted(REDUCTION_TECHNIQUES)
            raise ValueError(f"Unknown reduction method '{self.method}'. Choose from {valid}.")

    def fit_transform(
        self,
        data: Preprocessed | tuple[np.ndarray, np.ndarray],
        y: np.ndarray | None = None,
    ) -> Preprocessed | tuple[np.ndarray, np.ndarray]:
        """Fit reduction on training data and return the reduced data container."""
        return self.reduce(data, y=y).data


def reduce_dataset(
    data: Preprocessed | tuple[np.ndarray, np.ndarray],
    y: np.ndarray | None = None,
    method: str = "drop3",
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    **kwargs: Any,
) -> ReductionResult:
    """Convenience function to reduce a dataset using any supported method.

    Example:
        >>> result = reduce_dataset(train, method="drop3", distance="euclidean", k=3)
        >>> print(f"Retained {result.n_retained}/{result.n_original} instances")
        >>> reduced_train = result.data
    """
    reducer = InstanceReduction(
        method=method,
        distance=distance,
        voting=voting,
        **kwargs,
    )
    return reducer.reduce(data, y=y)


REDUCTION_TECHNIQUES: dict[str, Callable[..., ReductionResult]] = {
    "fcnn": fcnn,
    "fcnn1": fcnn,
    "fcnn2": lambda data, **kw: fcnn(data, variant="fcnn2", **kw),
    "fcnn3": lambda data, **kw: fcnn(data, variant="fcnn3", **kw),
    "fcnn4": lambda data, **kw: fcnn(data, variant="fcnn4", **kw),
    "eenth": eenth,
    "wilson_th": wilson_th,
    "wilson_prob": wilson_prob,
    "drop": drop,
    "drop1": drop1,
    "drop2": drop2,
    "drop3": drop3,
    "drop4": drop4,
    "drop5": drop5,
}
