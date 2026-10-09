"""Utilities for running experiments across the predefined cross-validation folds."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import config
from preprocessing import Preprocessed, preprocess_fold


@dataclass(frozen=True)
class ClassificationResult:
    """Predictions and number of training instances retained by a classifier."""

    predictions: list[int]
    storage: int


Classifier = Callable[[Preprocessed, Preprocessed], ClassificationResult]


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


Params = dict[str, object]


def format_params(params: Params | None) -> str:
    """Serialize parameters with sorted keys so equal configurations give equal strings."""
    if not params:
        return ""
    return ";".join(f"{key}={params[key]}" for key in sorted(params.keys()))


def run_fold(
    dataset: str,
    fold: int,
    classifier: Classifier,
    model_name: str,
    params: Params | None = None,
) -> ExperimentResult:
    """Preprocess one fold, run a classifier and collect its metrics."""
    train, test, _ = preprocess_fold(dataset, fold)

    start = perf_counter()
    result = classifier(train, test)
    elapsed = perf_counter() - start

    # SVM and RF from sklearn return a np.ndarray.
    predictions = [int(p) for p in result.predictions]

    if len(predictions) != len(test.y_codes):
        raise ValueError("Classifier returned a different number of predictions than test samples")

    if result.storage < 0 or result.storage > len(train.y):
        raise ValueError("Classifier storage must be between 0 and the number of training samples")

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
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "fold",
                "model",
                "params",
                "accuracy",
                "n_correct",
                "n_incorrect",
                "time",
                "storage",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "dataset": result.dataset,
                    "fold": result.fold,
                    "model": result.model,
                    "params": result.params,
                    "accuracy": result.accuracy,
                    "n_correct": result.n_correct,
                    "n_incorrect": result.n_incorrect,
                    "time": result.time,
                    "storage": result.storage,
                }
            )
