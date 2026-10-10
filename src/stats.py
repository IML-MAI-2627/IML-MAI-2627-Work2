"""Statistical comparison of classifiers following Demšar (2006).

Every test works on a score table: one row per fold (the blocks) and one column
per configuration (the classifiers). This is the c_i^j matrix of Demšar (2006),
Section 1, with the 10 folds of one dataset in place of different datasets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

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
HIGHER_IS_BETTER = {"accuracy": True, "time": False, "storage": False}


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman test over N folds (blocks) and k configurations."""

    n_blocks: int
    n_configs: int
    mean_ranks: pd.Series
    chi2: float
    chi2_p_value: float
    iman_davenport: float
    iman_davenport_p_value: float


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


def rank_table(table: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Rank the configurations inside each fold: 1 = best, ties get the average rank.

    Demšar (2006).
    """
    return table.rank(axis=1, ascending=not higher_is_better, method="average")


def mean_ranks(table: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Average rank R_j of each configuration over all folds, best (lowest) first.

    Demšar (2006)
    """
    return rank_table(table, higher_is_better).mean().sort_values()


def friedman_test(table: pd.DataFrame, higher_is_better: bool = True) -> FriedmanResult:
    """Friedman chi-square and the Iman-Davenport F statistic."""
    n_blocks, k = table.shape
    if k < 3:
        raise ValueError("Friedman needs at least 3 configurations, use Wilcoxon for 2")
    if n_blocks < 2:
        raise ValueError("Friedman needs at least 2 folds")

    ranks = mean_ranks(table, higher_is_better)

    factor = 12 * n_blocks / (k * (k + 1))
    bracket = (ranks**2).sum() - k * (k + 1) ** 2 / 4
    chi2 = factor * bracket
    chi2_p_value = sp_stats.chi2.sf(chi2, df=k - 1)

    denominator = n_blocks * (k - 1) - chi2
    if denominator <= 1e-12:
        iman_davenport = math.inf
    else:
        iman_davenport = (n_blocks - 1) * chi2 / denominator
    iman_davenport_p_value = sp_stats.f.sf(iman_davenport, k - 1, (k - 1) * (n_blocks - 1))

    return FriedmanResult(
        n_blocks=n_blocks,
        n_configs=k,
        mean_ranks=ranks,
        chi2=float(chi2),
        chi2_p_value=float(chi2_p_value),
        iman_davenport=float(iman_davenport),
        iman_davenport_p_value=float(iman_davenport_p_value),
    )
