"""Statistical comparison of classifiers following Demšar (2006).

Every test works on a score table: one row per fold (the blocks) and one column
per configuration (the classifiers). This is the c_i^ĵ matrix of Demšar (2006),
Section 1, with the 10 folds of one dataset in place of different datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = (
    "dataset",
    "fold",
    "model",
    "params",
    "accuracy",
    "n_correct",
    "n_incorrect",
    "time",
    "storage",
)
METRICS = ("accuracy", "time", "storage")


def load_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read one or more result CSVs written by the experiment runner into one DataFrame."""
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    if not frames:
        raise ValueError(f"No results found in {paths}")

    results = pd.concat(frames, ignore_index=True)
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Result files are missing columns: {sorted(missing)}")
    return results


def config_name(model: str, params: str) -> str:
    """One label per configuration: the model plus its parameter string."""
    return f"{model}[{params}]" if params else model


def score_table(
    results: pd.DataFrame,
    dataset: str,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Folds x configurations table of one metric for the one dataset (Demšar's c_i^j)."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")

    data = results[results["dataset"] == dataset]
    if data.empty:
        raise ValueError(f"No results for dataset {dataset!r}")

    data = data.assign(
        config=[
            config_name(model, params)
            for model, params in zip(data["model"], data["params"], strict=True)
        ]
    )

    if data.duplicated(["config", "fold"]).any():
        raise ValueError("Some configuration has the same fold more than once.")

    table = data.pivot(index="fold", columns="config", values=metric).sort_index()

    if table.isna().any().any():
        incomplete = table.columns[table.isna().any()].tolist()
        raise ValueError(f"Configurations without results on every fold: {incomplete}")
    return table
