"""Project-wide constants and helpers: paths, seed and fold file names."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

SEED = 42
N_FOLDS = 10
DATASETS = ("splice", "vowel")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int = SEED) -> None:
    """Fix the random seeds so that experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def fold_paths(dataset: str, fold: int) -> tuple[Path, Path]:
    """Return the (train, test) ARFF paths for a dataset and a fold index (0-9).

    Expected layout: datasets/<dataset>/<dataset>.fold.00000<k>.{train,test}.arff
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"Fold must be between 0 and {N_FOLDS - 1}, got {fold}")
    stem = f"{dataset}.fold.{fold:06d}"
    base = DATASETS_DIR / dataset
    return base / f"{stem}.train.arff", base / f"{stem}.test.arff"
