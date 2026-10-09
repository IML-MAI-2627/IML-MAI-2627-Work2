"""Run a majority-class baseline over the predefined 10 folds.

Run from the project root, for example:
    python experiments/run_baseline.py --dataset vowel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make src/ importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from experiment_runner import ClassificationResult, run_experiment, save_results  # noqa: E402
from preprocessing import Preprocessed  # noqa: E402


def majority_classifier(train: Preprocessed, test: Preprocessed) -> ClassificationResult:
    """Predict the most frequent training class; ties use the smallest class code."""
    majority_class = int(np.bincount(train.y_codes).argmax())
    return ClassificationResult(
        predictions=[majority_class] * len(test.y_codes),
        storage=len(train.y),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=config.DATASETS, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="CSV destination (default: results/baseline_<dataset>.csv)",
    )
    args = parser.parse_args()

    results = run_experiment(
        dataset=args.dataset,
        classifier=majority_classifier,
        model_name="majority_baseline",
        params={"vote": "majority_train_class"},
    )
    output = args.out or config.RESULTS_DIR / f"baseline_{args.dataset}.csv"
    save_results(results, output)
    try:
        display_path = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output
    print(f"Saved {len(results)} folds to {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
