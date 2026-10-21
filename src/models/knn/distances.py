"""Distance functions for the kNN, own code with numpy (no sklearn, no scipy.spatial).

Every distance receives the preprocessed ``train`` and ``test`` folds and returns the full
(n_test, n_train) matrix that ``knn_base.nearest_neighbors`` expects. Each metric chooses its
own representation: ``to_onehot()`` for the numeric-only metrics, ``X_num`` / ``X_nom`` for
mixed-type ones. Register new metrics in ``DISTANCES`` so they can be selected by name.
"""

from __future__ import annotations

import numpy as np

from models.knn.knn_utils import DistanceFunction
from preprocessing import Preprocessed


def _nominal_mismatches(train: Preprocessed, test: Preprocessed) -> np.ndarray:
    """(n_test, n_train) count of nominal attributes whose values differ.

    The matches come from a product of 0/1 one-hot matrices, which is exact and avoids an
    (n_test, n_train, n_onehot) array (~2 GB on splice).
    """
    n_num, n_nom = train.X_num.shape[1], train.X_nom.shape[1]
    # (n_test, n_train) with number of matches in each position
    matches = test.to_onehot()[:, n_num:] @ train.to_onehot()[:, n_num:].T
    # (n_test, n_train) with number of mismatched
    return n_nom - matches


def _triu_indices(data: Preprocessed) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangle index pairs (i, j) for i < j."""
    n = len(data.y_codes)
    return np.triu_indices(n, k=1)


def _fill_symmetric(n: int, i: np.ndarray, j: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Reconstruct an (n, n) symmetric distance matrix from upper-triangle values."""
    D = np.zeros((n, n), dtype=float)
    D[i, j] = values
    D[j, i] = values
    return D


def _nominal_mismatches_symmetric(data: Preprocessed, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Count mismatched nominal attributes for symmetric pairs (i, j) without inflating one-hot."""
    if data.X_nom.shape[1] == 0:
        return np.zeros(len(i), dtype=float)
    return np.sum(data.X_nom[i] != data.X_nom[j], axis=1).astype(float)


def _relative_differences_symmetric(data: Preprocessed, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Symmetric relative differences |a - b| / (|a| + |b|) for index pairs."""
    a = data.X_num[i]
    b = data.X_num[j]
    diff = np.abs(a - b)
    denom = np.abs(a) + np.abs(b)
    return np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)


def _relative_differences(train: Preprocessed, test: Preprocessed) -> np.ndarray:
    """(n_test, n_train, n_num) terms |x - y| / (|x| + |y|) of the numeric attributes.

    Equal to the slide's |x - y| / |x + y| for our non-negative, [0, 1]-scaled data. A 0/0
    term (both values 0) counts as 0: equal values add no distance.
    """
    a = test.X_num[:, None, :]
    b = train.X_num[None, :, :]
    diff = np.abs(a - b)
    denom = np.abs(a) + np.abs(b)
    # Divide only where denom > 0; elsewhere the term stays 0 (and numpy gives no warning).
    return np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)


def euclidean(train: Preprocessed, test: Preprocessed | None = None) -> np.ndarray:
    """Minkowski r=2 on the one-hot representation (all feature weights equal to 1).

    If ``test`` is omitted (or identical to ``train``), uses upper-triangle indexing
    (``np.triu_indices``) to compute the symmetric (n, n) distance matrix with 50% fewer
    arithmetic operations and reduced intermediate memory.
    """
    if test is None or test is train:
        # i = [0, 0, 0, 0...] j = [1, 2, 3, 4...]
        i, j = _triu_indices(train)
        if len(i) == 0:
            return np.zeros((len(train.y_codes), len(train.y_codes)), dtype=float)
        diff = train.X_num[i] - train.X_num[j]
        numeric = np.sum(diff**2, axis=1)
        mismatches = _nominal_mismatches_symmetric(train, i, j)
        dists = np.sqrt(numeric + 2.0 * mismatches)
        return _fill_symmetric(len(train.y_codes), i, j, dists)

    # Numeric part: exact squared differences, so equal rows give exactly 0 and ties stay ties.
    numeric = np.sum((test.X_num[:, None, :] - train.X_num[None, :, :]) ** 2, axis=2)

    # Nominal part: two one-hot rows of one attribute differ in 0 or 2 columns, so each
    # different value adds 2 to the squared sum.
    mismatches = _nominal_mismatches(train, test)

    return np.sqrt(numeric + 2.0 * mismatches)


def manhattan(train: Preprocessed, test: Preprocessed | None = None) -> np.ndarray:
    """Minkowski r=1 on the one-hot representation (all feature weights equal to 1).

    If ``test`` is omitted (or identical to ``train``), computes symmetric pairwise distances
    using ``np.triu_indices``.
    """
    if test is None or test is train:
        i, j = _triu_indices(train)
        if len(i) == 0:
            return np.zeros((len(train.y_codes), len(train.y_codes)), dtype=float)
        diff = np.abs(train.X_num[i] - train.X_num[j])
        numeric = np.sum(diff, axis=1)
        mismatches = _nominal_mismatches_symmetric(train, i, j)
        dists = numeric + 2.0 * mismatches
        return _fill_symmetric(len(train.y_codes), i, j, dists)

    # Numeric part: absolute differences instead of squared ones.
    numeric = np.sum(np.abs(test.X_num[:, None, :] - train.X_num[None, :, :]), axis=2)

    # Nominal part: each different value flips two one-hot columns, |1 - 0| = 1 each.
    mismatches = _nominal_mismatches(train, test)

    return numeric + 2.0 * mismatches


def canberra(train: Preprocessed, test: Preprocessed | None = None) -> np.ndarray:
    """Canberra distance (Wilson & Martinez, 1997) on the one-hot representation.

    If ``test`` is omitted (or identical to ``train``), computes symmetric pairwise distances
    using ``np.triu_indices``.
    """
    if test is None or test is train:
        i, j = _triu_indices(train)
        if len(i) == 0:
            return np.zeros((len(train.y_codes), len(train.y_codes)), dtype=float)
        rel_diff = _relative_differences_symmetric(train, i, j)
        numeric = rel_diff.sum(axis=1)
        mismatches = _nominal_mismatches_symmetric(train, i, j)
        dists = numeric + 2.0 * mismatches
        return _fill_symmetric(len(train.y_codes), i, j, dists)

    numeric = _relative_differences(train, test).sum(axis=2)
    # Each different nominal value flips two one-hot columns, |1 - 0| / (1 + 0) = 1 each.
    return numeric + 2.0 * _nominal_mismatches(train, test)


def clark(train: Preprocessed, test: Preprocessed | None = None) -> np.ndarray:
    """Clark distance as written in the course slides (no square root) on the one-hot
    representation. The usual definition takes the square root, which is monotonic, so the
    k nearest neighbours are the same either way; only distance-weighted votes change.

    If ``test`` is omitted (or identical to ``train``), computes symmetric pairwise distances
    using ``np.triu_indices``.
    """
    if test is None or test is train:
        i, j = _triu_indices(train)
        if len(i) == 0:
            return np.zeros((len(train.y_codes), len(train.y_codes)), dtype=float)
        rel_diff = _relative_differences_symmetric(train, i, j)
        numeric = (rel_diff**2).sum(axis=1)
        mismatches = _nominal_mismatches_symmetric(train, i, j)
        dists = numeric + 2.0 * mismatches
        return _fill_symmetric(len(train.y_codes), i, j, dists)

    numeric = (_relative_differences(train, test) ** 2).sum(axis=2)
    # Each different nominal value flips two one-hot columns, (1 / 1)^2 = 1 each.
    return numeric + 2.0 * _nominal_mismatches(train, test)


def heom(train: Preprocessed, test: Preprocessed | None = None) -> np.ndarray:
    """Heterogeneous Euclidean-Overlap Metric (Wilson & Martinez, 1997, section 2.3).

    Numeric attributes: |x - y| / range; our data is already min-max scaled with the train
    range, so the range is 1. Nominal attributes: overlap, 0 if equal and 1 if different.
    Unknown values would count as 1, but the preprocessing has already imputed them.
    Unlike ``euclidean`` on one-hot (2 per different value), a different value adds 1.

    If ``test`` is omitted (or identical to ``train``), computes symmetric pairwise distances
    using ``np.triu_indices``.
    """
    if test is None or test is train:
        i, j = _triu_indices(train)
        if len(i) == 0:
            return np.zeros((len(train.y_codes), len(train.y_codes)), dtype=float)
        diff = train.X_num[i] - train.X_num[j]
        numeric = np.sum(diff**2, axis=1)
        mismatches = _nominal_mismatches_symmetric(train, i, j)
        dists = np.sqrt(numeric + mismatches)
        return _fill_symmetric(len(train.y_codes), i, j, dists)

    numeric = np.sum((test.X_num[:, None, :] - train.X_num[None, :, :]) ** 2, axis=2)
    mismatches = _nominal_mismatches(train, test)
    return np.sqrt(numeric + mismatches)


DISTANCES: dict[str, DistanceFunction] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "canberra": canberra,
    "clark": clark,
    "heom": heom,
}
