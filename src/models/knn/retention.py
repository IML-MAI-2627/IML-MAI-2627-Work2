"""Retention policies for the kNN, own code (course slides, session 2, slide 22).

Every policy receives the predicted and the true class code of a solved test case and
says whether to store it in the case base (with its true class). Register new policies
in ``RETENTION`` so they can be selected by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

RetentionFunction = Callable[[int, int], bool]


@dataclass(frozen=True)
class OblivionByGoodness:
    """Oblivion by Goodness retention policy configuration and updater.

    Every case starts at ``initial`` goodness. After each query, only its retrieved
    neighbours update: G <- G + alpha * (r - G), with r = 1 if the neighbour's class equals
    the query's true class and 0 otherwise. If goodness drops below threshold, it is forgotten.
    """

    initial: float = 0.5
    alpha: float = 0.2
    threshold: float = 0.5

    def init_goodness(self, n_cases: int) -> np.ndarray:
        """Initialize goodness scores for all cases."""
        return np.full(n_cases, self.initial, dtype=float)

    def update_neighbor(
        self,
        neighbor_idx: int,
        is_correct: bool,
        goodness: np.ndarray,
        active: np.ndarray,
        min_cases: int,
    ) -> None:
        """Update goodness score of a retrieved neighbor and forget it if below threshold."""
        reward = 1.0 if is_correct else 0.0
        goodness[neighbor_idx] += self.alpha * (reward - goodness[neighbor_idx])
        # Forget it, but never leave fewer than min_cases in the case base
        if goodness[neighbor_idx] < self.threshold and active.sum() > min_cases:
            active[neighbor_idx] = False


def never_retain(prediction: int, label: int) -> bool:
    """NR: the case base is always the training set."""
    return False


def always_retain(prediction: int, label: int) -> bool:
    """AR: every solved test case is stored."""
    return True


def different_class(prediction: int, label: int) -> bool:
    """DF: store only the cases solved incorrectly (like IB2)."""
    return prediction != label


RETENTION: dict[str, RetentionFunction | OblivionByGoodness] = {
    "never": never_retain,
    "always": always_retain,
    "different_class": different_class,
    "oblivion_by_goodness": OblivionByGoodness(),
}
