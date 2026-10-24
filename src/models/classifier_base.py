from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from experiment_runner import ClassificationResult
from preprocessing import Preprocessed


class BaseClassifier(ABC):
    # Set to True to also return per-class scores (for ROC / PR curves). Off by default so
    # the timed experiments do not pay for them.
    return_scores: bool = False
    # Seconds of instance reduction in the last prepare_data call (0 without reduction), and
    # the seconds actually spent in that call (smaller on a reduction-cache hit).
    reduction_time: float = 0.0
    reduction_wall_time: float = 0.0
    # Folder where reduced training sets are stored and reused (None: no cache). Set it on the
    # instance, e.g. ``model.reduction_cache_dir = Path("results/gridsearch/reduction_cache")``.
    reduction_cache_dir: Path | None = None
    # Training (fit) time and inference (predict) time in seconds.
    fit_time: float = 0.0
    predict_time: float = 0.0

    def __init__(self, params: dict[str, object]):
        self.params = params
        self.parse_params()  # Forces subclasses to validate the given parameters.

    @abstractmethod
    def parse_params(self):
        """Override in subclasses to cast (ex. int(k)) and validate."""
        pass

    @abstractmethod
    def prepare_data(self, train: Preprocessed, test: Preprocessed) -> Any:
        """Prepare the data for the model fit and predict (including instance reduction)."""
        pass

    def reduce_train(self, train: Preprocessed) -> Preprocessed:
        """Apply the configured instance reduction (if any) and record its time."""
        self.reduction_time = self.reduction_wall_time = 0.0
        reducer = getattr(self, "reduction_instance", None)
        if reducer is None or reducer.method == "none":
            return train
        from models.knn.reduction import reduce_with_cache

        train, self.reduction_time, self.reduction_wall_time = reduce_with_cache(
            reducer, train, self.reduction_cache_dir
        )
        return train

    @abstractmethod
    def fit(self, train: Preprocessed) -> Any:
        """Fit the model on the preprocessed training set."""
        pass

    @abstractmethod
    def predict(self, test: Preprocessed) -> ClassificationResult:
        """Predict labels on the preprocessed test set."""
        pass

    @abstractmethod
    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Principal fit and predict logic."""
        pass


def full_class_scores(
    raw: np.ndarray, model_classes: np.ndarray, n_classes: int, fill: float
) -> np.ndarray:
    """Put the columns of a sklearn score matrix at their class codes: (n, n_classes).

    A reduced training set may lose a class; that column gets ``fill`` (a score no class
    present can lose to).
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:  # binary decision_function: score of the second class
        raw = np.column_stack([-raw, raw])
    scores = np.full((raw.shape[0], n_classes), fill, dtype=float)
    scores[:, np.asarray(model_classes, dtype=int)] = raw
    return scores
