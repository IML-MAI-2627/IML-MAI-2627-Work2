"""Shared test helpers."""

import numpy as np
import pytest

from arff_parser import NOMINAL, NUMERIC

# Small synthetic layout: two numeric and two nominal attributes
NAMES = ("a", "b", "color", "size")
TYPES = (NUMERIC, NUMERIC, NOMINAL, NOMINAL)
NOMINAL_VALUES = {"color": ("red", "green", "blue"), "size": ("S", "L")}
CLASS_VALUES = ("no", "yes")


def _make_data(cls, rows, y):
    """Build a TrainMatrix/TestMatrix from rows like [1.0, 5.0, "red", "S"].

    None means a missing value ('?'), stored like the parser does: NaN for numeric,
    None for nominal, and True in the missing mask.
    """
    n, d = len(rows), len(NAMES)
    X = np.empty((n, d), dtype=object)
    missing = np.zeros((n, d), dtype=bool)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            missing[i, j] = value is None
            if value is None and TYPES[j] == NUMERIC:
                value = float("nan")
            X[i, j] = value
    return cls(
        relation="synthetic",
        attribute_names=NAMES,
        attribute_types=TYPES,
        nominal_values=dict(NOMINAL_VALUES),
        X=X,
        missing=missing,
        y=np.array(y, dtype=object),
        class_name="class",
        class_values=CLASS_VALUES,
    )


@pytest.fixture
def make_data():
    return _make_data
