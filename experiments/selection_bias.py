"""Leave-one-fold-out estimate of the selection bias of the grid search (double dipping).

The grid picks the configuration with the best mean accuracy over the 10 folds and reports that
same mean, so the choice and the score use the same folds. For each fold f this script picks the
best configuration on the other 9 folds (same rule as the grid: accuracy, then time, then name)
and scores it on f. The mean of those 10 held-out accuracies never selects on the fold it scores;
its gap with the reported mean estimates the bias. Base configurations only (no reduction).

Run from the project root:  python experiments/selection_bias.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from experiments.grid_search import base_rows, load_dataset, summarize  # noqa: E402

MODELS = {"knn": "kNN", "svm": "SVM", "rf": "RF"}


def leave_one_fold_out(rows: pd.DataFrame) -> dict:
    """Reported best mean, leave-one-fold-out estimate and bias for one model's rows.

    ``rows`` has one row per (config, fold) with the columns ``summarize`` needs.
    """
    summary = summarize(rows)
    winner = summary["config"].iloc[0]
    reported = float(summary["accuracy"].iloc[0])
    folds = sorted(rows["fold"].unique())
    chosen, held_out = [], []
    for fold in folds:
        best = summarize(rows[rows["fold"] != fold])["config"].iloc[0]
        score = rows.loc[(rows["config"] == best) & (rows["fold"] == fold), "accuracy"]
        chosen.append(best)
        held_out.append(float(score.iloc[0]))
    estimate = sum(held_out) / len(held_out)
    return {
        "configurations": int(rows["config"].nunique()),
        "reported": reported,
        "leave_one_fold_out": estimate,
        "bias": reported - estimate,
        "distinct_chosen": len(set(chosen)),
        "rounds_with_winner": sum(c == winner for c in chosen),
    }


def selection_bias(results: Path) -> pd.DataFrame:
    """One row per (dataset, model)."""
    table = []
    for dataset in config.DATASETS:
        df = load_dataset(results, dataset)
        for key, name in MODELS.items():
            table.append(
                {"dataset": dataset, "model": name, **leave_one_fold_out(base_rows(df, key))}
            )
    return pd.DataFrame(table)


def markdown(table: pd.DataFrame) -> str:
    header = (
        "| Dataset | Model | Configurations | Best mean over the 10 folds (reported) "
        "| Leave-one-fold-out estimate | Estimated bias | Distinct winners in the 10 rounds "
        "| Rounds that chose the reported winner |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {r.dataset} | {r.model} | {r.configurations} | {r.reported:.4f} "
        f"| {r.leave_one_fold_out:.4f} | {r.bias:+.4f} | {r.distinct_chosen} "
        f"| {r.rounds_with_winner} |"
        for r in table.itertuples()
    ]
    return "# Selection bias (leave-one-fold-out)\n\n" + header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR / "gridsearch")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "selection_bias")
    args = parser.parse_args(argv)

    table = selection_bias(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "selection_bias.csv", index=False, float_format="%.6f")
    text = markdown(table)
    (args.out / "selection_bias.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {args.out / 'selection_bias.md'}")


if __name__ == "__main__":
    main()
