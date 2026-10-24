"""Instance reduction techniques: FCNN, ENNTH, DROP, and DEL."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from models.knn.distances import DISTANCES
from models.knn.knn_utils import (
    DistanceFunction,
    Neighbor,
    VotingFunction,
    check_k,
    classify_from_neighbors,
    compute_distance_matrix,
    kNNAlgorithm,
    subset_preprocessed,
)
from models.knn.voting import VOTING
from preprocessing import Preprocessed


@dataclass(frozen=True)
class ReductionResult:
    """Stores the retained indices and resulting reduced dataset."""

    indices: np.ndarray
    mask: np.ndarray
    n_original: int
    n_retained: int
    reduction_rate: float
    data: Preprocessed


def _wrap_result(train: Preprocessed, indices: np.ndarray) -> ReductionResult:
    """Bundle retained indices and subsetted preprocessed data into ReductionResult."""
    n_orig = len(train.y_codes)
    sorted_idx = np.sort(np.asarray(indices, dtype=np.int64))
    mask = np.zeros(n_orig, dtype=bool)
    mask[sorted_idx] = True
    n_ret = len(sorted_idx)
    rate = 1.0 - (n_ret / n_orig) if n_orig > 0 else 0.0

    return ReductionResult(
        indices=sorted_idx,
        mask=mask,
        n_original=n_orig,
        n_retained=n_ret,
        reduction_rate=rate,
        data=subset_preprocessed(train, sorted_idx),
    )


def resolve_distance(distance: str | Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Look up distance metric from name or return callable."""
    if callable(distance):
        return distance
    name = str(distance).lower()
    if name not in DISTANCES:
        raise ValueError(f"Unknown distance '{distance}'. Choose from {sorted(DISTANCES)}.")
    return DISTANCES[name]


def resolve_voting(
    voting: str | Callable[[Sequence[Neighbor]], int],
) -> Callable[[Sequence[Neighbor]], int]:
    """Look up voting policy from name or return callable."""
    if callable(voting):
        return voting
    name = str(voting).lower()
    if name not in VOTING:
        raise ValueError(f"Unknown voting '{voting}'. Choose from {sorted(VOTING)}.")
    return VOTING[name]


# ---------------------------------------------------------------------------
# Fast Condensed Nearest Neighbor (FCNN)
# ---------------------------------------------------------------------------


def fcnn(
    train: Preprocessed,
    version: int = 1,
    k: int = 1,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Fast Condensed Nearest Neighbor rule with Voronoi enemies and medoids."""
    n = len(train.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    if version not in (1, 2, 3, 4):
        raise ValueError(f"FCNN version must be 1, 2, 3, or 4 (got {version})")
    k = check_k(k, n)

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    # grab distance matrix or calculate it if missing
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(train, dist_fn)
    labels = train.y_codes
    classes = np.unique(labels)

    # find the medoid: exemplar that minimizes sum of squared distances to same-class points
    def get_class_medoid(subset_indices: np.ndarray | list[int]) -> int:
        sub = np.asarray(subset_indices, dtype=np.int64)
        if len(sub) == 1:
            return int(sub[0])
        sub_D = D[np.ix_(sub, sub)]
        scores = np.sum(sub_D**2, axis=1)
        best_local = np.argmin(scores)
        return int(sub[best_local])

    # start by picking medoids as our seeds
    if version in (3, 4):
        class_counts = {c: int(np.sum(labels == c)) for c in classes}
        most_pop_class = max(class_counts, key=class_counts.get)
        initial_seeds = [get_class_medoid(np.where(labels == most_pop_class)[0])]
    else:
        initial_seeds = [get_class_medoid(np.where(labels == c)[0]) for c in classes]

    S: set[int] = set(initial_seeds)
    delta_S: list[int] = sorted(initial_seeds)

    nearest = np.full(n, -1, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float64)

    for p in S:
        nearest[p] = p
        min_dist[p] = 0.0

    # keep iterating while new prototypes keep entering S
    while len(delta_S) > 0:
        S.update(delta_S)
        rem = np.array([i for i in range(n) if i not in S], dtype=np.int64)
        if len(rem) == 0:
            break

        # update distances to current prototypes
        delta_arr = np.array(delta_S, dtype=np.int64)
        delta_dists = D[np.ix_(rem, delta_arr)]
        best_delta_col = np.argmin(delta_dists, axis=1)
        best_delta_dist = delta_dists[np.arange(len(rem)), best_delta_col]
        best_delta_p = delta_arr[best_delta_col]

        closer_mask = best_delta_dist < min_dist[rem]
        closer_rem = rem[closer_mask]
        nearest[closer_rem] = best_delta_p[closer_mask]
        min_dist[closer_rem] = best_delta_dist[closer_mask]

        # let's look for voronoi enemies misclassified by S
        voren: dict[int, list[int]] = {p: [] for p in S}

        if k == 1:
            for q in rem:
                p_star = nearest[q]
                if labels[q] != labels[p_star]:
                    voren[p_star].append(q)
        else:
            # check k-NN consistency using the passed voting scheme
            S_list = np.array(sorted(S), dtype=np.int64)
            k_eff = min(k, len(S_list))
            sub_D = D[np.ix_(rem, S_list)]
            nb_order = np.argpartition(sub_D, k_eff - 1, axis=1)[:, :k_eff]
            for row_idx, q in enumerate(rem):
                nb_indices = S_list[nb_order[row_idx]]
                pred_label = classify_from_neighbors(q, nb_indices, D, labels, voting_fn)
                if labels[q] != pred_label:
                    p_star = nearest[q]
                    voren[p_star].append(q)

        # now pick who gets added next based on the version
        if version == 1:
            rep: dict[int, int] = {}
            for p, enemies in voren.items():
                if enemies:
                    best_enemy = enemies[np.argmin(D[p, enemies])]
                    rep[p] = best_enemy
            delta_S = sorted(set(rep.values()))

        elif version == 2:
            rep = {}
            for p, enemies in voren.items():
                if enemies:
                    enemy_classes = np.unique(labels[enemies])
                    c_medoids = [
                        get_class_medoid([e for e in enemies if labels[e] == c])
                        for c in enemy_classes
                    ]
                    best_medoid = min(c_medoids, key=lambda m: (D[p, m], m))
                    rep[p] = best_medoid
            delta_S = sorted(set(rep.values()))

        elif version == 3:
            max_enemies_p = max(voren.keys(), key=lambda p: (len(voren[p]), -p))
            if len(voren[max_enemies_p]) > 0:
                enemies = voren[max_enemies_p]
                best_enemy = enemies[np.argmin(D[max_enemies_p, enemies])]
                delta_S = [best_enemy]
            else:
                delta_S = []

        elif version == 4:
            max_enemies_p = max(voren.keys(), key=lambda p: (len(voren[p]), -p))
            if len(voren[max_enemies_p]) > 0:
                enemies = voren[max_enemies_p]
                enemy_classes = np.unique(labels[enemies])
                c_medoids = [
                    get_class_medoid([e for e in enemies if labels[e] == c]) for c in enemy_classes
                ]
                best_medoid = min(c_medoids, key=lambda m: (D[max_enemies_p, m], m))
                delta_S = [best_medoid]
            else:
                delta_S = []

        # stop if nothing new was found
        if not delta_S or all(p in S for p in delta_S):
            break

    return _wrap_result(train, np.array(sorted(S), dtype=np.int64))


# ---------------------------------------------------------------------------
# Edited Nearest Neighbors with Threshold (ENNTH)
# ---------------------------------------------------------------------------


def ennth(
    train: Preprocessed,
    k: int = 3,
    threshold: float = 0.0,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Filter out noisy or low-confidence border instances using k-NN posteriors."""
    n = len(train.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    # The class posteriors below have one column per class, indexed by class code.
    present = set(np.unique(train.y_codes).tolist())
    missing = [name for code, name in enumerate(train.class_values) if code not in present]
    if missing:
        raise ValueError(
            f"EENTh needs every class in the training data; missing classes: {missing}"
        )
    k = check_k(k)

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(train, dist_fn)
    labels = train.y_codes
    classes = np.unique(labels)
    n_classes = len(classes)

    # find k nearest neighbors excluding self
    k_eff = min(k, n - 1)
    if k_eff == 0:
        return _wrap_result(train, np.arange(n))

    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)

    order = np.argsort(D_no_diag, axis=1, kind="stable")[:, :k_eff]
    nb_dists = np.take_along_axis(D_no_diag, order, axis=1)
    nb_labels = labels[order]

    # inverse distance weights: w_j = 1 / (1 + dist)
    weights = 1.0 / (1.0 + nb_dists)
    total_weights = np.sum(weights, axis=1, keepdims=True)

    probs = np.zeros((n, n_classes), dtype=np.float64)
    for j in range(k_eff):
        np.add.at(probs, (np.arange(n), nb_labels[:, j]), weights[:, j])

    # normalize posteriors
    probs = np.divide(probs, total_weights, out=probs, where=total_weights > 0)

    best_class_idx = np.argmax(probs, axis=1)
    pred_class = classes[best_class_idx]
    max_prob = np.max(probs, axis=1)

    # break posterior probability ties with voting
    is_tied = np.sum(np.isclose(probs, max_prob[:, None]), axis=1) > 1
    if np.any(is_tied):
        for i in np.flatnonzero(is_tied):
            pred_class[i] = classify_from_neighbors(i, order[i], D, labels, voting_fn)

    # keep instance if correctly classified (and confidence exceeds threshold)
    correct_mask = pred_class == labels
    if threshold is not None and threshold > 0.0:
        keep_mask = correct_mask & (max_prob > threshold)
    else:
        keep_mask = correct_mask

    retained_indices = np.flatnonzero(keep_mask)

    # don't ever return an empty set
    if len(retained_indices) == 0:
        fallback = []
        for c in classes:
            c_mask = np.where(labels == c)[0]
            if len(c_mask) > 0:
                fallback.append(c_mask[np.argmax(max_prob[c_mask])])
        retained_indices = np.unique(fallback)

    return _wrap_result(train, retained_indices)


ENNTH = ennth


# ---------------------------------------------------------------------------
# Decremental Reduction Optimization Procedure (DROP) and DEL
# ---------------------------------------------------------------------------


def _compute_log_fact(n: int) -> np.ndarray:
    """Precompute log factorials ln(k!) for k = 0..n using cumulative sum."""
    log_fact = np.zeros(n + 1, dtype=np.float64)
    if n > 0:
        log_fact[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))
    return log_fact


def _logstar(log2_z: float) -> float:
    """Sum of positive iterated base-2 logarithms: log*(z) = log2(z) + log2(log2(z)) + ..."""
    if log2_z <= 0.0:
        return 0.0
    total = log2_z
    curr = np.log2(log2_z)
    while curr > 0.0:
        total += curr
        curr = np.log2(curr)
    return total


def _encoding_length_F(a: int, b: int, log_fact: np.ndarray) -> float:
    """Compute F(a, b) = log*(sum_{j=0}^a C(b, j)) = log*(sum_{j=0}^a (b!/(j!(b-j)!))
    numerically stably in log-space."""
    if b <= 0 or a <= 0:
        return 0.0
    a = min(a, b)
    if a == b:
        return _logstar(float(b))
    j_vals = np.arange(0, a + 1)
    # log_fact
    ln_binom = log_fact[b] - log_fact[j_vals] - log_fact[b - j_vals]
    max_val = np.max(ln_binom)
    ln_sum = max_val + np.log(np.sum(np.exp(ln_binom - max_val)))
    log2_sum = ln_sum / np.log(2.0)
    return _logstar(log2_sum)


def encoding_length_cost(m: int, n: int, x: int, num_classes: int, log_fact: np.ndarray) -> float:
    """Compute total encoding length cost:
    Cost(m, n, x) = F(m, n) + m*log2(C) + F(x, n - m) + x*log2(C - 1).
    """
    if num_classes <= 1:
        term_m_class = 0.0
        term_x_class = 0.0
    else:
        term_m_class = m * np.log2(num_classes)
        term_x_class = x * np.log2(num_classes - 1) if num_classes > 2 else 0.0

    f_m = _encoding_length_F(m, n, log_fact)
    f_x = _encoding_length_F(x, n - m, log_fact)
    return f_m + term_m_class + f_x + term_x_class


def drop(
    train: Preprocessed,
    version: int = 3,
    k: int = 3,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
    is_del: bool = False,
) -> ReductionResult:
    """Decremental prototype reduction (DROP version 1 through 5, or DEL)."""
    n = len(train.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    if not is_del and version not in (1, 2, 3, 4, 5):
        raise ValueError(f"DROP version must be 1, 2, 3, 4, or 5 (got {version})")
    k = check_k(k)

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(train, dist_fn)
    labels = train.y_codes
    classes = np.unique(labels)
    num_classes = len(classes)

    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)

    m_curr = n
    x_curr = 0
    curr_cost = 0.0
    log_fact = None
    if is_del:
        log_fact = _compute_log_fact(n)
        curr_cost = encoding_length_cost(m_curr, n, x_curr, num_classes, log_fact)

    # Initial subset S
    if is_del or version in (3, 4):
        k_eff = min(k, n - 1)
        enn_order = np.argsort(D_no_diag, axis=1, kind="stable")[:, :k_eff]
        misclassified: list[int] = []
        for i in range(n):
            pred = classify_from_neighbors(i, enn_order[i], D, labels, voting_fn)
            if pred != labels[i]:
                misclassified.append(i)

        if is_del:
            # Noise filtering pass for DEL: remove instance if misclassified by its k NNs
            # AND removing it does not increase encoding length cost.
            S_set = set(range(n))
            assert log_fact is not None
            for i in misclassified:
                if sum(1 for q in S_set if labels[q] == labels[i]) <= 1:
                    continue
                new_m = len(S_set) - 1
                new_x = x_curr + 1
                new_cost = encoding_length_cost(new_m, n, new_x, num_classes, log_fact)
                if new_cost <= curr_cost:
                    S_set.remove(i)
                    m_curr = new_m
                    x_curr = new_x
                    curr_cost = new_cost
        elif version == 3:
            # DROP3 drops all misclassified right away
            S_set = set(i for i in range(n) if i not in misclassified)
        else:
            # DROP4 only drops a noisy instance if it doesn't hurt associates
            S_set = set(range(n))
            for i in misclassified:
                if len(S_set) <= k + 1:
                    break
                with_c = 0
                without_c = 0
                for a in range(n):
                    if a == i:
                        continue
                    a_cands = [p for p in S_set if p != a]
                    if len(a_cands) < k:
                        continue
                    a_cands_order = sorted(a_cands, key=lambda p: (D[a, p], p))
                    top_k_with = a_cands_order[:k]
                    if i not in top_k_with:
                        continue
                    if classify_from_neighbors(a, top_k_with, D, labels, voting_fn) == labels[a]:
                        with_c += 1
                    top_k_without = [p for p in a_cands_order if p != i][:k]
                    if classify_from_neighbors(a, top_k_without, D, labels, voting_fn) == labels[a]:
                        without_c += 1
                if without_c >= with_c:
                    S_set.remove(i)
    else:
        S_set = set(range(n))

    # ensure every class has at least one representative medoid
    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_in_s = [p for p in S_set if labels[p] == c]
        if not c_in_s:
            c_all = np.where(labels == c)[0]
            sub_D = D[np.ix_(c_all, c_all)]
            best = c_all[np.argmin(np.sum(sub_D**2, axis=1))]
            S_set.add(int(best))

    def compute_enemy_distances(curr_s: set[int]) -> dict[int, float]:
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
        nonlocal m_curr, x_curr, curr_cost
        assoc_universe = list(curr_s) if track_in_s_only else list(range(n))

        # find k+1 nearest neighbors in curr_s for everyone in assoc_universe
        neighbors: dict[int, list[int]] = {}
        for a in assoc_universe:
            cands = [p for p in curr_s if p != a]
            if len(cands) <= k:
                neighbors[a] = sorted(cands, key=lambda p: (D[a, p], p))
            else:
                cands_sorted = sorted(cands, key=lambda p: (D[a, p], p))[: k + 1]
                neighbors[a] = cands_sorted

        # map who has who as their neighbor
        associates: dict[int, set[int]] = {p: set() for p in curr_s}
        for a in assoc_universe:
            for p in neighbors[a][:k]:
                if p in associates:
                    associates[p].add(a)

        removed_count = 0

        for p in evaluation_order:
            if p not in curr_s:
                continue
            if len(curr_s) <= k + 1:
                break
            p_class = labels[p]
            class_count_in_s = sum(1 for q in curr_s if labels[q] == p_class)
            if class_count_in_s <= 1:
                continue

            p_assoc = list(associates[p])

            with_corr = 0
            without_corr = 0
            can_remove = True

            # check if dropping p hurts its associates
            for a in p_assoc:
                curr_nb = neighbors[a]
                if classify_from_neighbors(a, curr_nb[:k], D, labels, voting_fn) == labels[a]:
                    with_corr += 1

                nb_without = [idx for idx in curr_nb if idx != p][:k]
                if len(nb_without) < k:
                    can_remove = False
                    break
                if classify_from_neighbors(a, nb_without, D, labels, voting_fn) == labels[a]:
                    without_corr += 1

            if not can_remove:
                continue

            if is_del:
                assert log_fact is not None
                delta_x = with_corr - without_corr
                p_cands = sorted([q for q in curr_s if q != p], key=lambda q: (D[p, q], q))[:k]
                if len(p_cands) >= k:
                    p_pred = classify_from_neighbors(p, p_cands, D, labels, voting_fn)
                    if p_pred != labels[p]:
                        delta_x += 1
                else:
                    delta_x += 1

                new_m = len(curr_s) - 1
                new_x = max(0, x_curr + delta_x)
                new_cost = encoding_length_cost(new_m, n, new_x, num_classes, log_fact)
                should_remove = new_cost <= curr_cost
            else:
                should_remove = without_corr >= with_corr

            if should_remove:
                # safe to drop p
                curr_s.remove(p)
                removed_count += 1
                del associates[p]
                if is_del:
                    m_curr = new_m
                    x_curr = new_x
                    curr_cost = new_cost

                # update neighbor and associate links
                for a in p_assoc:
                    neighbors[a] = [idx for idx in neighbors[a] if idx != p]
                    curr_nb_set = set(neighbors[a])
                    new_cands = [q for q in curr_s if q != a and q not in curr_nb_set]
                    if new_cands:
                        best_new = min(new_cands, key=lambda q: (D[a, q], q))
                        neighbors[a].append(best_new)

                    if len(neighbors[a]) >= k:
                        kth_nb = neighbors[a][k - 1]
                        if kth_nb in associates:
                            associates[kth_nb].add(a)

                # in DROP1, eliminate p from neighbors' associate lists since p is out of S
                if track_in_s_only:
                    if p in neighbors:
                        for nbr in neighbors[p]:
                            if nbr in associates:
                                associates[nbr].discard(p)

        return removed_count

    if is_del:
        edist = compute_enemy_distances(S_set)
        order = sorted(S_set, key=lambda p: (edist[p], p), reverse=True)
        run_removal_pass(S_set, order, track_in_s_only=False)

    elif version == 1:
        run_removal_pass(S_set, list(S_set), track_in_s_only=True)

    elif version in (2, 3, 4):
        edist = compute_enemy_distances(S_set)
        order = sorted(S_set, key=lambda p: (edist[p], p), reverse=True)
        run_removal_pass(S_set, order, track_in_s_only=False)

    elif version == 5:
        edist = compute_enemy_distances(S_set)
        order = sorted(S_set, key=lambda p: (edist[p], p))
        run_removal_pass(S_set, order, track_in_s_only=False)

        while True:
            edist = compute_enemy_distances(S_set)
            order = sorted(S_set, key=lambda p: (-edist[p], p))
            n_rem = run_removal_pass(S_set, order, track_in_s_only=False)
            if n_rem == 0:
                break

    return _wrap_result(train, np.array(sorted(S_set), dtype=np.int64))


def del_reduction(
    train: Preprocessed,
    k: int = 3,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Decremental Encoding Length (DEL) reduction technique (Wilson & Martinez 2000)."""
    return drop(
        train,
        version=3,
        k=k,
        distance=distance,
        voting=voting,
        distance_matrix=distance_matrix,
        is_del=True,
    )


# ---------------------------------------------------------------------------
# Object-Oriented Hierarchy and Polymorphic Factory
# ---------------------------------------------------------------------------


class InstanceReduction(ABC):
    """Abstract base class and polymorphic factory for instance reduction techniques.

    Used by: KNN.parse_params(), SVM.parse_params(), RandomForest.parse_params(),
    and BaseClassifier to validate reduction hyperparameters and prune inactive ones.
    """

    method: str = "none"
    k: int = 3
    distance: str | DistanceFunction = "euclidean"
    voting: str | VotingFunction = "majority"

    def __new__(
        cls,
        params_or_method: dict[str, Any] | str = "drop",
        *args: Any,
        **kwargs: Any,
    ) -> InstanceReduction:
        if cls is not InstanceReduction:
            return super().__new__(cls)

        if isinstance(params_or_method, dict):
            raw_method = str(params_or_method.get("reduction", "none")).lower()
        else:
            raw_method = str(params_or_method).lower()

        if raw_method not in VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction '{raw_method}'. Must be one of: {sorted(VALID_REDUCTIONS)}."
            )

        target_cls = REDUCTION_CLASSES[raw_method]
        return super().__new__(target_cls)

    @abstractmethod
    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        """Run reduction on preprocessed training data."""
        pass

    def fit_transform(self, train: Preprocessed) -> Preprocessed:
        """Return the filtered Preprocessed object."""
        return self.reduce(train).data


class NoReduction(InstanceReduction):
    """No instance reduction: retains all original instances."""

    method = "none"

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "none",
        *,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 3,
        **kwargs: Any,
    ) -> None:
        if isinstance(params_or_method, dict):
            params = params_or_method
            params["reduction"] = "none"
            disallowed = [p for p in ("reduction_k", "version", "threshold") if p in params]
            if disallowed:
                raise ValueError(
                    f"Parameters {disallowed} are not allowed when reduction is 'none'"
                )
            raw_k = params.get("reduction_k", params.get("k", k))
            try:
                k_val = int(str(raw_k))
            except ValueError:
                raise ValueError(f"Invalid k '{raw_k}'. Must be a positive integer.") from None
            self.k = check_k(k_val)
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)
        else:
            disallowed = [p for p in ("reduction_k", "version", "threshold") if p in kwargs]
            if disallowed:
                raise ValueError(
                    f"Parameters {disallowed} are not allowed when reduction is 'none'"
                )
            self.k = check_k(int(str(k)))
            self.distance = distance
            self.voting = voting

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        return _wrap_result(train, np.arange(len(train.y_codes)))


class FCNNReduction(InstanceReduction):
    """Fast Condensed Nearest Neighbor (FCNN) reduction."""

    method = "fcnn"

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "fcnn",
        *,
        version: int = 1,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 1,
        **kwargs: Any,
    ) -> None:
        if isinstance(params_or_method, dict):
            params = params_or_method
            params["reduction"] = "fcnn"
            if "threshold" in params:
                raise ValueError(
                    "Unknown parameter 'threshold' for FCNN (FCNN only accepts 'version')"
                )
            raw_ver = params.get("version", 1)
            try:
                ver = int(str(raw_ver))
            except ValueError:
                raise ValueError(f"Invalid version '{raw_ver}'. Must be an integer.") from None
            if ver not in (1, 2, 3, 4):
                raise ValueError(f"FCNN version must be 1, 2, 3, or 4 (got {ver})")
            self.version = ver
            params["version"] = self.version
            raw_k = params.get("reduction_k", params.get("k", k))
            try:
                k_val = int(str(raw_k))
            except ValueError:
                raise ValueError(f"Invalid k '{raw_k}'. Must be a positive integer.") from None
            self.k = check_k(k_val)
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)
        else:
            if "threshold" in kwargs:
                raise ValueError(
                    "Unknown parameter 'threshold' for FCNN (FCNN only accepts 'version')"
                )
            if version not in (1, 2, 3, 4):
                raise ValueError(f"FCNN version must be 1, 2, 3, or 4 (got {version})")
            self.version = version
            self.k = check_k(int(str(k)))
            self.distance = distance
            self.voting = voting

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        return fcnn(
            train,
            version=self.version,
            distance=self.distance,
            voting=self.voting,
            k=self.k,
            distance_matrix=distance_matrix,
        )


class DROPReduction(InstanceReduction):
    """Decremental Reduction Optimization Procedure (DROP1..5)."""

    method = "drop"

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "drop",
        *,
        version: int = 3,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 3,
        **kwargs: Any,
    ) -> None:
        if isinstance(params_or_method, dict):
            params = params_or_method
            params["reduction"] = "drop"
            if "threshold" in params:
                raise ValueError(
                    "Unknown parameter 'threshold' for DROP (DROP only accepts 'version')"
                )
            raw_ver = params.get("version", 3)
            try:
                ver = int(str(raw_ver))
            except ValueError:
                raise ValueError(f"Invalid version '{raw_ver}'. Must be an integer.") from None
            if ver not in (1, 2, 3, 4, 5):
                raise ValueError(f"DROP version must be 1, 2, 3, 4, or 5 (got {ver})")
            self.version = ver
            params["version"] = self.version
            raw_k = params.get("reduction_k", params.get("k", k))
            try:
                k_val = int(str(raw_k))
            except ValueError:
                raise ValueError(f"Invalid k '{raw_k}'. Must be a positive integer.") from None
            self.k = check_k(k_val)
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)
        else:
            if "threshold" in kwargs:
                raise ValueError(
                    "Unknown parameter 'threshold' for DROP (DROP only accepts 'version')"
                )
            if version not in (1, 2, 3, 4, 5):
                raise ValueError(f"DROP version must be 1, 2, 3, 4, or 5 (got {version})")
            self.version = version
            self.k = check_k(int(str(k)))
            self.distance = distance
            self.voting = voting

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        return drop(
            train,
            version=self.version,
            distance=self.distance,
            voting=self.voting,
            k=self.k,
            distance_matrix=distance_matrix,
        )


class ENNTHReduction(InstanceReduction):
    """Edited Nearest Neighbors with Threshold (ENNTH)."""

    method = "ennth"

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "ennth",
        *,
        threshold: float = 0.0,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 3,
        **kwargs: Any,
    ) -> None:
        if isinstance(params_or_method, dict):
            params = params_or_method
            params["reduction"] = "ennth"
            if "version" in params:
                raise ValueError(
                    "Unknown parameter 'version' for ENNTH (ENNTH only accepts 'threshold')"
                )
            raw_th = params.get("threshold", 0.0)
            try:
                th = float(str(raw_th))
            except ValueError:
                raise ValueError(f"Invalid threshold '{raw_th}'. Must be a float.") from None
            if th < 0.0:
                raise ValueError(f"Threshold must be non-negative, got {th}")
            self.threshold = th
            params["threshold"] = self.threshold
            raw_k = params.get("reduction_k", params.get("k", k))
            try:
                k_val = int(str(raw_k))
            except ValueError:
                raise ValueError(f"Invalid k '{raw_k}'. Must be a positive integer.") from None
            self.k = check_k(k_val)
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)
        else:
            if "version" in kwargs:
                raise ValueError(
                    "Unknown parameter 'version' for ENNTH (ENNTH only accepts 'threshold')"
                )
            th = float(threshold)
            if th < 0.0:
                raise ValueError(f"Threshold must be non-negative, got {th}")
            self.threshold = th
            self.k = check_k(int(str(k)))
            self.distance = distance
            self.voting = voting

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        return ennth(
            train,
            distance=self.distance,
            voting=self.voting,
            k=self.k,
            threshold=self.threshold,
            distance_matrix=distance_matrix,
        )


class DELReduction(InstanceReduction):
    """Decremental Encoding Length (DEL) reduction technique."""

    method = "del"

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "del",
        *,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 3,
        **kwargs: Any,
    ) -> None:
        if isinstance(params_or_method, dict):
            params = params_or_method
            params["reduction"] = "del"
            disallowed = [p for p in ("version", "threshold") if p in params]
            if disallowed:
                raise ValueError(f"Parameters {disallowed} are not allowed for DEL")
            raw_k = params.get("reduction_k", params.get("k", k))
            try:
                k_val = int(str(raw_k))
            except ValueError:
                raise ValueError(f"Invalid k '{raw_k}'. Must be a positive integer.") from None
            self.k = check_k(k_val)
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)
        else:
            disallowed = [p for p in ("version", "threshold") if p in kwargs]
            if disallowed:
                raise ValueError(f"Parameters {disallowed} are not allowed for DEL")
            self.k = check_k(int(str(k)))
            self.distance = distance
            self.voting = voting

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        return del_reduction(
            train,
            distance=self.distance,
            voting=self.voting,
            k=self.k,
            distance_matrix=distance_matrix,
        )


REDUCTION_PARAM_KEYS = {
    "reduction",
    "reduction_k",
    "version",
    "threshold",
    "distance",
    "voting",
    "reduction_distance",
    "reduction_voting",
}


def parse_reduction_params(params: dict[str, Any], is_knn: bool = False) -> InstanceReduction:
    """Validate and configure instance reduction hyperparameters.

    For kNN: uses 'k', 'distance', 'voting' from kNN parameters.
    Rejects 'reduction_k', 'reduction_distance', 'reduction_voting'.
    For non-kNN models: uses 'reduction_k' (default 3), 'reduction_distance'/'distance',
    'reduction_voting'/'voting', and rejects 'k' if not a model parameter.
    """
    if is_knn:
        for forbidden in ("reduction_k", "reduction_distance", "reduction_voting"):
            if forbidden in params:
                base = forbidden.replace("reduction_", "")
                raise ValueError(
                    f"Unknown parameter '{forbidden}' for kNN "
                    f"(kNN uses '{base}' for both classification and reduction)"
                )

        if "reduction" not in params:
            reduction_params_passed = [p for p in ("version", "threshold") if p in params]
            if reduction_params_passed:
                raise ValueError(
                    f"Passing reduction parameters {reduction_params_passed} without specifying "
                    "the reduction type parameter '--reduction' is not allowed. "
                    "Please add the reduction type parameter '--reduction'."
                )
            return NoReduction()

        method = str(params["reduction"]).lower()
        if method not in VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction '{method}'. Must be one of: {sorted(VALID_REDUCTIONS)}."
            )

        red_dict: dict[str, Any] = {
            "reduction": method,
            "k": params.get("k", 3),
            "distance": params.get("distance", "euclidean"),
            "voting": params.get("voting", "majority"),
        }
        for opt_key in ("version", "threshold"):
            if opt_key in params:
                red_dict[opt_key] = params[opt_key]

        return InstanceReduction(red_dict)

    # For non-kNN models:
    if "k" in params:
        raise ValueError(
            "Unknown parameter 'k' for non-kNN model. "
            "Use 'reduction_k' to configure neighbor count for reduction."
        )

    if "reduction" not in params:
        reduction_params_passed = [
            p
            for p in (
                "reduction_k",
                "reduction_distance",
                "reduction_voting",
                "version",
                "threshold",
                "distance",
                "voting",
            )
            if p in params
        ]
        if reduction_params_passed:
            raise ValueError(
                f"Passing reduction parameters {reduction_params_passed} without specifying "
                "the reduction type parameter '--reduction' is not allowed. "
                "Please add the reduction type parameter '--reduction'."
            )
        return NoReduction()

    method = str(params["reduction"]).lower()
    if method not in VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{method}'. Must be one of: {sorted(VALID_REDUCTIONS)}."
        )

    if "reduction_k" in params:
        raw_rk = params["reduction_k"]
        try:
            rk = int(str(raw_rk))
        except ValueError:
            raise ValueError(
                f"Invalid reduction_k '{raw_rk}'. Must be a positive integer."
            ) from None
        red_k = check_k(rk)
    else:
        red_k = 3

    red_dist = params.get("reduction_distance", params.get("distance", "euclidean"))
    red_voting = params.get("reduction_voting", params.get("voting", "majority"))

    red_dict: dict[str, Any] = {
        "reduction": method,
        "k": red_k,
        "reduction_k": red_k,
        "distance": red_dist,
        "voting": red_voting,
    }
    for opt_key in ("version", "threshold"):
        if opt_key in params:
            red_dict[opt_key] = params[opt_key]

    reduction_instance = InstanceReduction(red_dict)

    if method != "none":
        params["reduction"] = method
        params["reduction_k"] = red_k
        if method in ("fcnn", "drop"):
            params["version"] = getattr(reduction_instance, "version", 1)
        elif method == "ennth":
            params["threshold"] = getattr(reduction_instance, "threshold", 0.0)

    return reduction_instance


def reduce_dataset(
    train: Preprocessed,
    method: str = "drop",
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    **kwargs: Any,
) -> ReductionResult:
    """Helper to reduce a dataset using the requested reduction method."""
    reducer = InstanceReduction(
        method,
        distance=distance,
        voting=voting,
        **kwargs,
    )
    return reducer.reduce(train)


REDUCTION_CLASSES: dict[str, type[InstanceReduction]] = {
    "none": NoReduction,
    "fcnn": FCNNReduction,
    "drop": DROPReduction,
    "ennth": ENNTHReduction,
    "del": DELReduction,
}

REDUCTION_TECHNIQUES: dict[str, Callable[..., ReductionResult]] = {
    "fcnn": fcnn,
    "ennth": ennth,
    "drop": drop,
    "del": del_reduction,
}

VALID_REDUCTIONS = {"none", "fcnn", "ennth", "drop", "del"}


# ---------------------------------------------------------------------------
# On-disk cache of reduced training sets
# ---------------------------------------------------------------------------
# Every technique is deterministic, so the same reducer on the same training fold always keeps
# the same instances. The grid search applies one reduction to kNN, SVM and RF; with the cache
# it is computed once. The caller passes the folder explicitly; without it nothing is cached.


def _cache_key(reducer: InstanceReduction, train: Preprocessed) -> str | None:
    """Hash of the reducer settings and the training data; None if not cacheable."""
    if callable(reducer.distance) or callable(reducer.voting):
        return None  # a custom function has no stable name
    settings = (
        reducer.method,
        reducer.k,
        str(reducer.distance).lower(),
        str(reducer.voting).lower(),
        getattr(reducer, "version", None),
        getattr(reducer, "threshold", None),
    )
    digest = hashlib.sha1(repr(settings).encode())
    for array in (train.X_num, train.X_nom, train.y_codes):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def reduce_with_cache(
    reducer: InstanceReduction,
    train: Preprocessed,
    cache_dir: str | Path | None = None,
) -> tuple[Preprocessed, float, float]:
    """Reduce ``train``; returns (reduced set, reduction seconds, seconds spent in this call).

    With ``cache_dir`` the reduced set is stored there and reused on later calls with the same
    reducer and training data. The reduction seconds are the ones measured when the reduction
    was actually computed, also on a cache hit, so the reported times do not depend on the cache.
    """
    start = perf_counter()
    key = _cache_key(reducer, train) if cache_dir is not None else None
    path = Path(cache_dir) / f"{key}.npz" if key is not None else None
    if path is not None and path.exists():
        cached = np.load(path)
        reduced = subset_preprocessed(train, cached["indices"])
        return reduced, float(cached["seconds"]), perf_counter() - start

    D = compute_distance_matrix(train, resolve_distance(reducer.distance))
    indices = reducer.reduce(train, distance_matrix=D).indices
    seconds = perf_counter() - start
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, indices=indices, seconds=seconds)
        tmp.replace(path)  # atomic: parallel workers never read a half-written file
    return subset_preprocessed(train, indices), seconds, seconds


def reduction_kNNAlgorithm(
    train: Preprocessed,
    test: Preprocessed,
    k: int,
    distance_function: str | DistanceFunction,
    voting_function: str | VotingFunction,
    reduction: str,
    retention_function: Callable[[int, int], bool] | None = None,
    **reduction_params: Any,
) -> list[int]:
    """kNN on a reduced training set: reduce ``train``, then run ``kNNAlgorithm``.

    Named as the assignment requires (``reduction_kNNAlgorithm(...)``). The reduction uses the
    same k, distance and voting as the kNN; ``reduction`` is ``"fcnn"``, ``"ennth"``, ``"drop"``
    or ``"del"`` and ``reduction_params`` its options (``version``, ``threshold``).
    """
    distance = resolve_distance(distance_function)
    voting = resolve_voting(voting_function)
    reducer = InstanceReduction(
        reduction, k=k, distance=distance, voting=voting, **reduction_params
    )
    reduced = reducer.reduce(train).data
    return kNNAlgorithm(reduced, test, k, distance, voting, retention_function)
