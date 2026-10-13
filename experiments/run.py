"""Run a majority-class baseline over the predefined 10 folds.

Run from the project root, for example:
    python experiments/run.py --dataset vowel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from experiment_runner import run_experiment, save_results  # noqa: E402
from models.majority_baseline import MajorityBaseline  # noqa: E402
from models.random_forest import RandomForest  # noqa: E402

CLASSIFIERS = {
    "baseline": (MajorityBaseline, "Majority Baseline"),
    "rf": (RandomForest, "Random Forest"),
    # "knn": (KNNClassifier, "K-Nearest Neighbors"),
    # "svm": (SVMClassifier, "Support Vector Machine"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=config.DATASETS, required=True)
    parser.add_argument("--classifier", choices=CLASSIFIERS.keys(), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="CSV destination (default: results/<classifier>_<dataset>.csv)",
    )

    # parse_known_args returns two things: known arguments and a list of the rest
    args, unknown_args = parser.parse_known_args()

    # We need to make a dictionary of the unknown parameters
    params = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]  # Drop the --
            # If the following hasn't --, then it is the value
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                params[key] = unknown_args[i + 1]
                i += 2
            else:
                # It is a boolean flag (ex: --verbose)
                params[key] = True
                i += 1
        else:
            raise ValueError("Two values given for a single parameter.")

    model_class = CLASSIFIERS[args.classifier][0]
    model_instance = model_class(params)

    results = run_experiment(
        dataset=args.dataset,
        classifier=model_instance.fit_predict,
        model_name=CLASSIFIERS[args.classifier][1],
        params=params,
    )
    output = args.out or config.RESULTS_DIR / f"{args.classifier}_{args.dataset}.csv"
    save_results(results, output)
    try:
        display_path = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output
    print(f"Saved {len(results)} folds to {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
