"""Preprocessing fitted on the TRAIN fold only and then applied to train and test.

Numeric attributes: missing -> train mean, then min-max to [0,1] with the train min/max.
Test values outside the train range are clipped to [0,1], so every distance term stays
bounded. A constant column (max == min) becomes 0.

Nominal attributes: missing -> train mode, then integer codes. The code of a value is its
position in the ARFF header (``nominal_values``), so codes are the same in every fold.
Each attribute stays as ONE column (needed for the HEOM overlap distance in the kNN).
``Preprocessed.to_onehot()`` gives the one-hot version for SVM and RF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arff_parser import NUMERIC, ArffData, TrainMatrix, load_fold

# Columns removed before learning.
# vowel "Train_or_Test" only records which half of the original UCI split the row came from.
# It does not describe the sound and is unrelated to our 10 folds, so it would only add
# noise to the distances.
# vowel "Speaker_Number" identifies who is speaking. Each speaker says every vowel equally
# often, so it says nothing about the class and it would pull neighbours towards the same
# speaker. "Sex" is kept. See docs/DECISIONS.md.
DEFAULT_DROP: dict[str, tuple[str, ...]] = {
    "splice": (),
    "vowel": ("Train_or_Test", "Speaker_Number"),
}


@dataclass(frozen=True)
class Preprocessed:
    """A preprocessed fold, ready for the classifiers."""

    X_num: np.ndarray  # (n, n_num) float, all values in [0,1]
    X_nom: np.ndarray  # (n, n_nom) int codes, one column per nominal attribute
    y: np.ndarray  # (n,) class labels, unchanged
    y_codes: np.ndarray  # (n,) int, position of the label in class_values
    num_names: tuple[str, ...]  # name of each column of X_num
    nom_names: tuple[str, ...]  # name of each column of X_nom
    nom_values: tuple[tuple[str, ...], ...]  # header values of each nominal column
    class_values: tuple[str, ...]

    @property
    def nom_cardinalities(self) -> tuple[int, ...]:
        """Number of possible values of each nominal column."""
        return tuple(len(values) for values in self.nom_values)

    def to_onehot(self) -> np.ndarray:
        """Numeric columns followed by one 0/1 column per nominal value (for SVM/RF)."""
        blocks = [self.X_num]
        for j, size in enumerate(self.nom_cardinalities):
            blocks.append(np.eye(size)[self.X_nom[:, j]])
        return np.hstack(blocks)

    def onehot_names(self) -> tuple[str, ...]:
        """Column names of ``to_onehot()``, e.g. 'Sex=Male'."""
        names = list(self.num_names)
        for name, values in zip(self.nom_names, self.nom_values, strict=True):
            names += [f"{name}={value}" for value in values]
        return tuple(names)


class Preprocessor:
    """Learns the statistics on the train fold (``fit``) and applies them (``transform``)."""

    def __init__(self, drop: tuple[str, ...] | list[str] = ()) -> None:
        self.drop = tuple(drop)
        self._fitted = False

    @classmethod
    def for_dataset(cls, dataset: str) -> Preprocessor:
        """Preprocessor with the default dropped columns of a dataset."""
        return cls(drop=DEFAULT_DROP.get(dataset, ()))

    def fit(self, train: TrainMatrix) -> Preprocessor:
        """Compute min/max/mean and mode from the train fold only."""
        unknown = [name for name in self.drop if name not in train.attribute_names]
        if unknown:
            raise ValueError(f"Columns to drop not in the data: {unknown}")

        # Remember the layout of the train fold to check the test fold later
        self.attribute_names_ = train.attribute_names
        self.attribute_types_ = train.attribute_types
        self.nominal_values_ = dict(train.nominal_values)
        self.class_values_ = train.class_values

        # Which columns we keep, split by type
        kept = [j for j, name in enumerate(train.attribute_names) if name not in self.drop]
        self.num_idx_ = [j for j in kept if train.attribute_types[j] == NUMERIC]
        self.nom_idx_ = [j for j in kept if train.attribute_types[j] != NUMERIC]
        self.num_names_ = tuple(train.attribute_names[j] for j in self.num_idx_)
        self.nom_names_ = tuple(train.attribute_names[j] for j in self.nom_idx_)
        self.nom_values_ = tuple(train.nominal_values[name] for name in self.nom_names_)

        # Numeric statistics, using only the values that are not missing
        self.min_ = np.zeros(len(self.num_idx_))
        self.max_ = np.zeros(len(self.num_idx_))
        self.mean_ = np.zeros(len(self.num_idx_))
        for k, j in enumerate(self.num_idx_):
            present = train.X[~train.missing[:, j], j].astype(np.float64)
            if present.size > 0:  # if everything is missing the stats stay at 0
                self.min_[k] = present.min()
                self.max_[k] = present.max()
                self.mean_[k] = present.mean()

        # Mode of each nominal column, using only the values that are not missing.
        # argmax returns the first maximum, so ties go to the first value in the header.
        self.mode_ = np.zeros(len(self.nom_idx_), dtype=np.int64)
        for k, j in enumerate(self.nom_idx_):
            codes, missing = self._encode_column(train, j, self.nom_values_[k])
            counts = np.bincount(codes[~missing], minlength=len(self.nom_values_[k]))
            self.mode_[k] = counts.argmax()

        self._fitted = True
        return self

    def transform(self, data: ArffData) -> Preprocessed:
        """Apply the train statistics to a fold (train or test). Nothing is learned here."""
        if not self._fitted:
            raise RuntimeError("Call fit() with the train fold before transform()")
        if (
            data.attribute_names != self.attribute_names_
            or data.attribute_types != self.attribute_types_
            or dict(data.nominal_values) != self.nominal_values_
            or data.class_values != self.class_values_
        ):
            raise ValueError("The data does not have the same attributes as the train fold")

        n = data.n_instances

        # Numeric: fill missing with the train mean, scale with the train range, clip
        X_num = np.zeros((n, len(self.num_idx_)))
        for k, j in enumerate(self.num_idx_):
            col = data.X[:, j].astype(np.float64)
            col[data.missing[:, j]] = self.mean_[k]
            width = self.max_[k] - self.min_[k]
            if width > 0:
                X_num[:, k] = np.clip((col - self.min_[k]) / width, 0.0, 1.0)
            # constant column in train: left as 0

        # Nominal: header position as code, missing filled with the train mode
        X_nom = np.zeros((n, len(self.nom_idx_)), dtype=np.int64)
        for k, j in enumerate(self.nom_idx_):
            codes, missing = self._encode_column(data, j, self.nom_values_[k])
            codes[missing] = self.mode_[k]
            X_nom[:, k] = codes

        # Class: kept as it is, plus an integer version
        class_code = {label: i for i, label in enumerate(self.class_values_)}
        y_codes = np.array([class_code[label] for label in data.y], dtype=np.int64)

        return Preprocessed(
            X_num=X_num,
            X_nom=X_nom,
            y=data.y.copy(),
            y_codes=y_codes,
            num_names=self.num_names_,
            nom_names=self.nom_names_,
            nom_values=self.nom_values_,
            class_values=self.class_values_,
        )

    def fit_transform(self, train: TrainMatrix) -> Preprocessed:
        return self.fit(train).transform(train)

    @staticmethod
    def _encode_column(
        data: ArffData, j: int, values: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integer codes of nominal column j (0 where missing) and its missing mask."""
        code = {value: i for i, value in enumerate(values)}
        missing = data.missing[:, j]
        codes = np.zeros(data.n_instances, dtype=np.int64)
        for i in np.flatnonzero(~missing):
            value = data.X[i, j]
            if value not in code:
                name = data.attribute_names[j]
                raise ValueError(f"Value {value!r} of {name!r} is not declared in the header")
            codes[i] = code[value]
        return codes, missing


def preprocess_fold(dataset: str, fold: int) -> tuple[Preprocessed, Preprocessed, Preprocessor]:
    """Load one fold, fit on its train part and return (train, test, preprocessor)."""
    train, test = load_fold(dataset, fold)
    preprocessor = Preprocessor.for_dataset(dataset)
    return preprocessor.fit_transform(train), preprocessor.transform(test), preprocessor
