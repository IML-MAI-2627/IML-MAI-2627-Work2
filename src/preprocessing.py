"""Preprocessing fitted on the TRAIN fold only and then applied to train and test.

Built with scikit-learn transformers (allowed for preprocessing, not for the kNN).

Numeric attributes: missing -> train mean (SimpleImputer), then min-max to [0,1] with the
train min/max (MinMaxScaler). Test values outside the train range are clipped to [0,1], so
every distance term stays bounded. A constant column (max == min) becomes 0.

Nominal attributes: missing -> train mode (SimpleImputer), then integer codes
(OrdinalEncoder). The code of a value is its position in the ARFF header
(``nominal_values``), so codes are the same in every fold.
Each attribute stays as ONE column (needed for the HEOM overlap distance in the kNN).
``Preprocessed.to_onehot()`` gives the one-hot version for SVM and RF.

Optional (off by default, only for the sensitivity analysis of splice): ``drop_duplicate_rows``
removes from the TRAIN fold the rows identical to an earlier one in every attribute and the class.
The test fold is never changed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

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

    def reduce(self, reducer: object) -> Preprocessed:
        """Apply an instance reduction technique to this preprocessed dataset."""
        if hasattr(reducer, "reduce"):
            return reducer.reduce(self).data
        return self


def reduce_preprocessed(train: Preprocessed, reducer: object) -> Preprocessed:
    """Apply instance reduction to a preprocessed training dataset."""
    if hasattr(reducer, "reduce"):
        return reducer.reduce(train).data
    return train


def remove_duplicate_rows(train: TrainMatrix) -> tuple[TrainMatrix, np.ndarray]:
    """Train fold without the rows identical to an earlier one; also the kept row indices.

    Two rows are identical when every attribute (missing cells included) and the class are equal.
    The first appearance is kept. Rows with the same attributes and different classes are all
    kept: they are label noise, not copies.
    """
    seen: set[tuple] = set()
    kept = []
    for i in range(train.n_instances):
        key = (
            tuple(None if train.missing[i, j] else train.X[i, j] for j in range(train.n_features)),
            train.y[i],
        )
        if key not in seen:
            seen.add(key)
            kept.append(i)
    kept = np.asarray(kept, dtype=np.int64)
    subset = replace(train, X=train.X[kept], missing=train.missing[kept], y=train.y[kept])
    return subset, kept


class Preprocessor:
    """Learns the statistics on the train fold (``fit``) and applies them (``transform``).

    With ``drop_duplicate_rows=True``, ``fit`` and ``fit_transform`` first remove the exact
    duplicates of the train fold (``remove_duplicate_rows``), so the statistics and the returned
    train come from the deduplicated rows. ``transform`` never removes rows, so the test fold is
    always complete.
    """

    def __init__(
        self,
        drop: tuple[str, ...] | list[str] = (),
        max_missing: float = 0.5,
        on_missing: str = "error",
        drop_duplicate_rows: bool = False,
    ) -> None:
        if not 0 <= max_missing <= 1:
            raise ValueError("max_missing must be between 0 and 1")
        if on_missing not in ("error", "drop"):
            raise ValueError("on_missing must be 'error' or 'drop'")
        self.drop = tuple(drop)
        self.max_missing = max_missing
        self.on_missing = on_missing
        self.drop_duplicate_rows = drop_duplicate_rows
        self._fitted = False

    @classmethod
    def for_dataset(cls, dataset: str, **kwargs) -> Preprocessor:
        """Preprocessor with the default dropped columns of a dataset."""
        return cls(drop=DEFAULT_DROP.get(dataset, ()), **kwargs)

    def fit(self, train: TrainMatrix) -> Preprocessor:
        """Fit the sklearn pipelines (min/max/mean and mode) on the train fold only."""
        return self._fit(self._training_rows(train))

    def _fit(self, train: TrainMatrix) -> Preprocessor:
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
        # Own rule: columns with too many missing train values (fraction > max_missing).
        # Measured on the train fold only, like every other statistic (no leakage).
        too_missing = [j for j in kept if train.missing[:, j].mean() > self.max_missing]
        too_missing_names = [train.attribute_names[j] for j in too_missing]
        if too_missing_names and self.on_missing == "error":
            raise ValueError(
                f"Columns with more than {self.max_missing:.0%} of train values missing: "
                f"{too_missing_names}. Add them to drop=... or use on_missing='drop'."
            )
        self.dropped_missing_ = tuple(too_missing_names)
        kept = [j for j in kept if j not in too_missing]

        self.num_idx_ = [j for j in kept if train.attribute_types[j] == NUMERIC]
        self.nom_idx_ = [j for j in kept if train.attribute_types[j] != NUMERIC]
        self.num_names_ = tuple(train.attribute_names[j] for j in self.num_idx_)
        self.nom_names_ = tuple(train.attribute_names[j] for j in self.nom_idx_)
        self.nom_values_ = tuple(train.nominal_values[name] for name in self.nom_names_)

        # Numeric: mean imputation, then min-max with clipping.
        # The imputer ignores the missing cells, and filling with the mean cannot move
        # the min or the max, so the three statistics come from the present values only.
        # keep_empty_features: a column with every value missing is filled with 0.
        self.min_ = np.zeros(0)
        self.max_ = np.zeros(0)
        self.mean_ = np.zeros(0)
        # Own rule: sklearn rejects an array with 0 columns (splice has no numeric ones)
        if self.num_idx_:
            self._num_pipeline = make_pipeline(
                SimpleImputer(strategy="mean", keep_empty_features=True),
                MinMaxScaler(clip=True),
            ).fit(self._numeric_block(train))
            imputer, scaler = self._num_pipeline[0], self._num_pipeline[1]
            self.mean_ = imputer.statistics_.copy()
            self.min_ = scaler.data_min_.copy()
            self.max_ = scaler.data_max_.copy()

        # Nominal: mode imputation on the text values, then header position as code.
        # Mode ties: "most_frequent" returns the smallest value, i.e. alphabetical order.
        self.mode_ = np.zeros(0, dtype=np.int64)
        # Own rule: sklearn rejects an array with 0 columns (all nominal columns dropped)
        if self.nom_idx_:
            # Own rule: with no value at all there is no mode, and sklearn would silently
            # drop the column, so we stop with a clear error
            empty = [train.attribute_names[j] for j in self.nom_idx_ if train.missing[:, j].all()]
            if empty:
                raise ValueError(f"Nominal columns with every train value missing: {empty}")
            self._nom_pipeline = make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(categories=[list(v) for v in self.nom_values_]),
            ).fit(self._nominal_block(train))
            imputer, encoder = self._nom_pipeline[0], self._nom_pipeline[1]
            # the mode as a code, like the values in X_nom
            self.mode_ = encoder.transform(imputer.statistics_.reshape(1, -1))[0].astype(np.int64)

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
        if self.num_idx_:  # own rule: sklearn rejects 0 columns
            X_num = self._num_pipeline.transform(self._numeric_block(data))
            # Own rule: MinMaxScaler only shifts a constant column, and a different test
            # value would become 1 after clipping, so we set the column to 0
            X_num[:, self.max_ == self.min_] = 0.0

        # Nominal: missing filled with the train mode, then header position as code.
        # A value that is not in the header makes the encoder raise ValueError.
        X_nom = np.zeros((n, len(self.nom_idx_)), dtype=np.int64)
        if self.nom_idx_:  # own rule: sklearn rejects 0 columns
            X_nom = self._nom_pipeline.transform(self._nominal_block(data)).astype(np.int64)

        # Class: kept as it is, plus an integer version.
        # Own rule: a row whose class is unknown ("?" or any value not in the header) cannot
        # be learned from or evaluated, so it is removed with a warning instead of stopping.
        class_code = {label: i for i, label in enumerate(self.class_values_)}
        known = np.array([label in class_code for label in data.y], dtype=bool)
        if not known.all():
            warnings.warn(f"Removed {(~known).sum()} rows with an unknown class", stacklevel=2)
        y_codes = np.array([class_code[label] for label in data.y[known]], dtype=np.int64)

        return Preprocessed(
            X_num=X_num[known],
            X_nom=X_nom[known],
            y=data.y[known],
            y_codes=y_codes,
            num_names=self.num_names_,
            nom_names=self.nom_names_,
            nom_values=self.nom_values_,
            class_values=self.class_values_,
        )

    def fit_transform(self, train: TrainMatrix) -> Preprocessed:
        train = self._training_rows(train)
        return self._fit(train).transform(train)

    def _training_rows(self, train: TrainMatrix) -> TrainMatrix:
        """The train fold, without its exact duplicates if ``drop_duplicate_rows``."""
        self.n_duplicates_removed_ = 0
        if not self.drop_duplicate_rows:
            return train
        deduplicated, kept = remove_duplicate_rows(train)
        self.n_duplicates_removed_ = train.n_instances - len(kept)
        return deduplicated

    def _numeric_block(self, data: ArffData) -> np.ndarray:
        """Kept numeric columns as floats, NaN where the value is missing."""
        block = data.X[:, self.num_idx_].astype(np.float64)
        block[data.missing[:, self.num_idx_]] = np.nan
        return block

    def _nominal_block(self, data: ArffData) -> np.ndarray:
        """Kept nominal columns as strings, NaN where missing (what SimpleImputer expects)."""
        block = data.X[:, self.nom_idx_].copy()
        block[data.missing[:, self.nom_idx_]] = np.nan
        return block


def preprocess_fold(dataset: str, fold: int) -> tuple[Preprocessed, Preprocessed, Preprocessor]:
    """Load one fold, fit on its train part and return (train, test, preprocessor)."""
    train, test = load_fold(dataset, fold)
    preprocessor = Preprocessor.for_dataset(dataset)
    return preprocessor.fit_transform(train), preprocessor.transform(test), preprocessor
