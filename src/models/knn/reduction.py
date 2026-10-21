"""Instance reduction techniques: FCNN, EENTh, and DROP."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
# Extended Edited Nearest Neighbors with Threshold (EENTh)
# ---------------------------------------------------------------------------


def eenth(
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


# ---------------------------------------------------------------------------
# Decremental Reduction Optimization Procedure (DROP)
# ---------------------------------------------------------------------------


def drop(
    train: Preprocessed,
    version: int = 3,
    k: int = 3,
    distance: str | Callable[..., np.ndarray] = "euclidean",
    voting: str | Callable[[Sequence[Neighbor]], int] = "majority",
    distance_matrix: np.ndarray | None = None,
) -> ReductionResult:
    """Decremental prototype reduction (DROP version 1 through 5)."""
    n = len(train.y_codes)
    if n == 0:
        raise ValueError("training data must not be empty")
    if version not in (1, 2, 3, 4, 5):
        raise ValueError(f"DROP version must be 1, 2, 3, 4, or 5 (got {version})")
    k = check_k(k)

    dist_fn = resolve_distance(distance)
    voting_fn = resolve_voting(voting)
    D = distance_matrix if distance_matrix is not None else compute_distance_matrix(train, dist_fn)
    labels = train.y_codes

    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)

    # Initial subset S
    if version in (3, 4):
        k_eff = min(k, n - 1)
        enn_order = np.argsort(D_no_diag, axis=1, kind="stable")[:, :k_eff]
        misclassified: list[int] = []
        for i in range(n):
            pred = classify_from_neighbors(i, enn_order[i], D, labels, voting_fn)
            if pred != labels[i]:
                misclassified.append(i)

        if version == 3:
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

            if can_remove and without_corr >= with_corr:
                # safe to drop p
                curr_s.remove(p)
                removed_count += 1
                del associates[p]

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

    if version == 1:
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


# ---------------------------------------------------------------------------
# Object-Oriented Wrapper and Registry
# ---------------------------------------------------------------------------


class InstanceReduction:
    """Instance reduction manager that validates and prunes configuration parameters.

    Used by: KNN.parse_params() to automatically validate reduction hyperparameters and
    prune inactive ones from model params.
    """

    def __init__(
        self,
        params_or_method: dict[str, Any] | str = "drop",
        *,
        distance: str | DistanceFunction = "euclidean",
        voting: str | VotingFunction = "majority",
        k: int = 3,
        version: int = 1,
        threshold: float = 0.0,
    ) -> None:
        if isinstance(params_or_method, dict):
            # Called with the KNN params dictionary: parse, validate, and prune
            params = params_or_method
            method = str(params.get("reduction", "none")).lower()
            if method not in VALID_REDUCTIONS:
                raise ValueError(
                    f"Invalid reduction '{method}'. Must be one of: {sorted(VALID_REDUCTIONS)}."
                )

            self.method = method
            params["reduction"] = self.method
            self.k = check_k(params.get("k", k))
            self.distance = params.get("distance", distance)
            self.voting = params.get("voting", voting)

            if self.method == "none":
                disallowed = [p for p in ("version", "threshold") if p in params]
                if disallowed:
                    raise ValueError(
                        f"Parameters {disallowed} are not allowed when reduction is 'none'"
                    )
                self.version = 1
                self.threshold = 0.0

            elif self.method == "fcnn":
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
                self.threshold = 0.0

            elif self.method == "drop":
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
                self.threshold = 0.0

            elif self.method == "eenth":
                if "version" in params:
                    raise ValueError(
                        "Unknown parameter 'version' for EENTh (EENTh only accepts 'threshold')"
                    )
                raw_th = params.get("threshold", threshold)
                try:
                    th = float(str(raw_th))
                except ValueError:
                    raise ValueError(f"Invalid threshold '{raw_th}'. Must be a float.") from None
                self.threshold = th
                params["threshold"] = self.threshold
                self.version = 1

        else:
            # Standalone instantiation: InstanceReduction("drop", version=3, ...)
            method = str(params_or_method).lower()
            if method not in VALID_REDUCTIONS:
                raise ValueError(
                    f"Unknown reduction method '{method}'. Choose from {sorted(VALID_REDUCTIONS)}."
                )
            self.method = method
            self.distance = distance
            self.voting = voting
            self.k = check_k(k)
            self.version = version
            self.threshold = threshold

    def reduce(
        self,
        train: Preprocessed,
        distance_matrix: np.ndarray | None = None,
    ) -> ReductionResult:
        """Run reduction on preprocessed training data."""
        if self.method == "none":
            return _wrap_result(train, np.arange(len(train.y_codes)))

        if self.method == "fcnn":
            return fcnn(
                train,
                version=self.version,
                distance=self.distance,
                voting=self.voting,
                k=self.k,
                distance_matrix=distance_matrix,
            )

        if self.method == "eenth":
            return eenth(
                train,
                distance=self.distance,
                voting=self.voting,
                k=self.k,
                threshold=self.threshold,
                distance_matrix=distance_matrix,
            )

        if self.method == "drop":
            return drop(
                train,
                version=self.version,
                distance=self.distance,
                voting=self.voting,
                k=self.k,
                distance_matrix=distance_matrix,
            )

        raise ValueError(f"Unknown reduction method '{self.method}'")

    def fit_transform(self, train: Preprocessed) -> Preprocessed:
        """Return the filtered Preprocessed object."""
        return self.reduce(train).data


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


REDUCTION_TECHNIQUES: dict[str, Callable[..., ReductionResult]] = {
    "fcnn": fcnn,
    "eenth": eenth,
    "drop": drop,
}

VALID_REDUCTIONS = {"none", "fcnn", "eenth", "drop"}
