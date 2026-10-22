"""Utilities for running experiments across the predefined cross-validation folds."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from time import perf_counter

import numpy as np

import config
from preprocessing import Preprocessed, preprocess_fold


@dataclass(frozen=True)
class ClassificationResult:
    """Predictions and number of training instances retained by a classifier.

    ``scores`` (n_test, n_classes) is optional: per-class scores for ROC / PR curves,
    filled only when the classifier is asked for them (``return_scores``).
    """

    predictions: list[int]
    storage: int
    scores: np.ndarray | None = None
    reduction_time: float = 0.0  # part of the timed call spent reducing the training set
    # Seconds of reduction actually spent in this call; smaller than reduction_time when the
    # reduced set came from the cache. None means equal to reduction_time.
    reduction_wall_time: float | None = None


Classifier = Callable[
    [
        Preprocessed,
        Preprocessed,
    ],
    ClassificationResult,
]


@dataclass(frozen=True)
class ExperimentResult:
    """Result of one classifier run on one predefined fold."""

    dataset: str
    fold: int
    model: str
    params: str
    accuracy: float
    n_correct: int
    n_incorrect: int
    time: float
    storage: float
    reduction_time: float
    macro_f1: float
    confusion_matrix: str  # JSON list of lists: rows = true class, columns = predicted


Params = dict[str, object]

# Common CSV schema of every experiment (one row per fold).
CSV_COLUMNS = (
    "dataset",
    "fold",
    "model",
    "params",
    "accuracy",
    "n_correct",
    "n_incorrect",
    "time",
    "storage",
    "reduction_time",
    "macro_f1",
    "confusion_matrix",
)


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) counts: row = true class code, column = predicted class code."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    """Unweighted mean of the per-class F1 = 2TP / (2TP + FP + FN).

    Averaged over the classes that appear in the test fold or in the predictions (the same
    rule as scikit-learn's default); a class that is never predicted gets F1 = 0.
    """
    tp = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)  # = 2TP + FP + FN
    present = denominator > 0
    if not present.any():
        return 0.0
    return float(np.mean(2 * tp[present] / denominator[present]))


def parse_confusion_matrix(text: str) -> np.ndarray:
    """Read back the ``confusion_matrix`` column of a results CSV."""
    return np.asarray(json.loads(text), dtype=np.int64)


def format_params(params: Params | None) -> str:
    """Serialize parameters with sorted keys so equal configurations give equal strings."""
    if not params:
        return ""
    return ";".join(f"{key}={params[key]}" for key in sorted(params.keys()))


@cache
def load_fold(dataset: str, fold: int):
    """Preprocess one fold once and reuse it for every configuration."""
    return preprocess_fold(dataset, fold)


def run_fold(
    dataset: str,
    fold: int,
    classifier: Classifier,
    model_name: str,
    params: Params | None = None,
) -> ExperimentResult:
    """Preprocess one fold, run a classifier and collect its metrics."""
    train, test, _ = load_fold(dataset, fold)

    start = perf_counter()
    result = classifier(train, test)
    elapsed = perf_counter() - start
    # A cached reduction is charged with its originally measured time, so `time` means the same
    # with or without the reduction cache.
    reduction_time = float(getattr(result, "reduction_time", 0.0))
    wall = getattr(result, "reduction_wall_time", None)
    if wall is not None:
        elapsed += reduction_time - float(wall)

    # SVM and RF from sklearn return a np.ndarray.
    predictions = list(result.predictions)
    for prediction in predictions:
        if isinstance(prediction, bool) or not isinstance(prediction, int | np.integer):
            raise ValueError("Classifier predictions must be integer class codes")
    predictions = [int(p) for p in predictions]

    if len(predictions) != len(test.y_codes):
        raise ValueError("Classifier returned a different number of predictions than test samples")

    # Retention policies can store solved test cases too, so the case base can grow up to
    # train + test (storage still cannot be negative or exceed every case seen).
    max_storage = len(train.y) + len(test.y)
    if result.storage < 0 or result.storage > max_storage:
        raise ValueError(
            "Classifier storage must be between 0 and the number of training plus test samples"
        )

    n_correct = sum(
        prediction == int(expected)
        for prediction, expected in zip(
            predictions,
            test.y_codes,
            strict=True,
        )
    )
    n_incorrect = len(predictions) - n_correct
    accuracy = n_correct / len(predictions) if len(predictions) > 0 else 0.0
    storage = 100.0 * result.storage / len(train.y)

    # Class order = position in the ARFF header (class_values), the same in every fold.
    n_classes = len(getattr(test, "class_values", ())) or (
        max([*predictions, *map(int, test.y_codes)], default=-1) + 1
    )
    matrix = confusion_matrix(test.y_codes, predictions, n_classes)

    return ExperimentResult(
        dataset=dataset,
        fold=fold,
        model=model_name,
        params=format_params(params),
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        time=elapsed,
        storage=storage,
        reduction_time=reduction_time,
        macro_f1=macro_f1(matrix),
        confusion_matrix=json.dumps(matrix.tolist(), separators=(",", ":")),
    )


def run_experiment(
    dataset: str,
    classifier: Classifier,
    model_name: str,
    params: Params | None = None,
) -> list[ExperimentResult]:
    """Run one classifier on all predefined folds of a dataset with a fixed seed."""
    config.set_seed()

    return [
        run_fold(
            dataset,
            fold,
            classifier,
            model_name,
            params,
        )
        for fold in range(config.N_FOLDS)
    ]


def save_results(
    results: list[ExperimentResult],
    path: str | Path,
) -> None:
    """Save experiment results using the common CSV schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({column: getattr(result, column) for column in CSV_COLUMNS})
