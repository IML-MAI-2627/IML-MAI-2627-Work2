"""Reads ARFF files with scipy.io.arff and returns TrainMatrix / TestMatrix.

X holds the features (numeric as float, nominal as str), missing marks the '?' values
and y holds the class labels. No normalization or imputation here (see preprocessing).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from scipy.io import arff

import config

NUMERIC = "numeric"
NOMINAL = "nominal"
_MISSING_NOMINAL = "?"

T = TypeVar("T", bound="ArffData")


class ArffParseError(ValueError):
    """Raised when an ARFF file is malformed or uses an unsupported feature."""


@dataclass(frozen=True)
class ArffData:
    """Parsed ARFF file: features, missing mask and class labels kept separately."""

    relation: str
    attribute_names: tuple[str, ...]
    attribute_types: tuple[str, ...]  # "numeric" | "nominal", one per feature
    nominal_values: dict[str, tuple[str, ...]]  # declared values of each nominal feature
    X: np.ndarray  # (n, d) object array, see module docstring
    missing: np.ndarray  # (n, d) bool, True where the value was '?'
    y: np.ndarray  # (n,) str class labels
    class_name: str
    class_values: tuple[str, ...]

    @property
    def n_instances(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def numeric_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.attribute_types) if t == NUMERIC]

    def nominal_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.attribute_types) if t == NOMINAL]

    def numeric_matrix(self) -> np.ndarray:
        """Numeric feature columns as a float64 array (NaN where missing)."""
        return self.X[:, self.numeric_indices()].astype(np.float64)


@dataclass(frozen=True)
class TrainMatrix(ArffData):
    """Training fold."""


@dataclass(frozen=True)
class TestMatrix(ArffData):
    """Test fold."""

    __test__ = False  # not a pytest test class


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_arff(path: str | Path, cls: type[T], class_index: int = -1) -> T:
    """Read an ARFF file into ``cls`` (TrainMatrix/TestMatrix). The class is the last
    attribute by default; use ``class_index`` to choose another one."""
    path = Path(path)
    try:
        # 1. Read the file with scipy (data = rows, meta = header with names and types)
        data, meta = arff.loadarff(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # scipy raises ParseArffError, ValueError, IndexError...
        raise ArffParseError(f"Cannot parse {path}: {exc}") from exc

    # 2. Validate the file: fail with a clear error instead of guessing
    names = list(meta.names())
    types = list(meta.types())
    if not names:
        raise ArffParseError(f"{path}: no attributes declared")
    if len(data) == 0:
        raise ArffParseError(f"{path}: no data rows")
    for name, typ in zip(names, types, strict=True):
        if typ not in (NUMERIC, NOMINAL):
            raise ArffParseError(f"{path}: attribute {name!r} has unsupported type {typ!r}")
    if not -len(names) <= class_index < len(names):
        raise ArffParseError(f"{path}: class_index {class_index} out of range for {len(names)}")
    class_index %= len(names)

    # 3. Take the class column: it must be nominal and have no '?'
    class_name = names[class_index]
    if types[class_index] != NOMINAL:
        raise ArffParseError(f"{path}: class attribute {class_name!r} must be nominal")

    class_col = [_decode(v) for v in data[class_name]]
    if _MISSING_NOMINAL in class_col:
        raise ArffParseError(f"{path}: missing class label ('?') in {class_name!r}")

    # 4. Features = every column except the class; prepare empty X and missing tables
    feat_names = [n for i, n in enumerate(names) if i != class_index]
    feat_types = [t for i, t in enumerate(types) if i != class_index]
    n, d = len(data), len(feat_names)
    X = np.empty((n, d), dtype=object)
    missing = np.zeros((n, d), dtype=bool)
    nominal_values: dict[str, tuple[str, ...]] = {}

    # 5. Fill X and missing column by column
    for j, (name, typ) in enumerate(zip(feat_names, feat_types, strict=True)):
        col = data[name]
        # numeric: float, NaN where the value was '?'
        if typ == NUMERIC:
            values = np.asarray(col, dtype=np.float64)
            missing[:, j] = np.isnan(values)
            X[:, j] = values.tolist()
        else:
            # nominal: str, None where the value was '?'
            nominal_values[name] = tuple(meta[name][1])
            decoded = [_decode(v) for v in col]
            missing[:, j] = [v == _MISSING_NOMINAL for v in decoded]
            X[:, j] = [None if m else v for v, m in zip(decoded, missing[:, j], strict=True)]

    # 6. Pack everything in a TrainMatrix / TestMatrix
    return cls(
        relation=meta.name,
        attribute_names=tuple(feat_names),
        attribute_types=tuple(feat_types),
        nominal_values=nominal_values,
        X=X,
        missing=missing,
        y=np.array(class_col, dtype=object),
        class_name=class_name,
        class_values=tuple(meta[class_name][1]),
    )


def load_fold(dataset: str, fold: int) -> tuple[TrainMatrix, TestMatrix]:
    """Load the predefined (train, test) pair of a dataset for fold 0-9."""
    train_path, test_path = config.fold_paths(dataset, fold)
    return load_arff(train_path, TrainMatrix), load_arff(test_path, TestMatrix)
