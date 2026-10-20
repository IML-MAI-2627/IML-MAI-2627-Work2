"""Retention policies for the kNN, own code (course slides, session 2, slide 22).

Every policy receives the predicted and the true class code of a solved test case and
says whether to store it in the case base (with its true class). Register new policies
in ``RETENTION`` so they can be selected by name.
"""

from __future__ import annotations

from models.knn.knn_base import OblivionByGoodness, RetentionFunction


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
