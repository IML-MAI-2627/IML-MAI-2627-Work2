"""Distance functions for the kNN, own code with numpy (no sklearn, no scipy.spatial).

Every distance receives the preprocessed ``train`` and ``test`` folds and returns the full
(n_test, n_train) matrix that ``knn_base.nearest_neighbors`` expects. Each metric chooses its
own representation: ``to_onehot()`` for the numeric-only metrics, ``X_num`` / ``X_nom`` for
mixed-type ones. Register new metrics in ``DISTANCES`` so they can be selected by name.
"""

from __future__ import annotations

import numpy as np

from knn_base import DistanceFunction
from preprocessing import Preprocessed


def euclidean(train: Preprocessed, test: Preprocessed) -> np.ndarray:
    """Minkowski r=2 on the one-hot representation (all feature weights equal to 1)."""
    # Numeric part: exact squared differences, so equal rows give exactly 0 and ties stay ties.
    numeric = np.sum((test.X_num[:, None, :] - train.X_num[None, :, :]) ** 2, axis=2)

    # Nominal part: two one-hot rows of one attribute differ in 0 or 2 columns, so its
    # squared contribution is 2 * (number of different values). The matches come from a
    # product of 0/1 matrices, which is exact, and it avoids an (n_test, n_train, n_onehot)
    # array (~2 GB on splice).
    n_num, n_nom = train.X_num.shape[1], train.X_nom.shape[1]
    matches = test.to_onehot()[:, n_num:] @ train.to_onehot()[:, n_num:].T
    mismatches = n_nom - matches

    return np.sqrt(numeric + 2.0 * mismatches)


DISTANCES: dict[str, DistanceFunction] = {
    "euclidean": euclidean,
}
