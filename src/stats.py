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
METRICS = ("accuracy", "macro_f1", "time", "storage")
HIGHER_IS_BETTER = {"accuracy": True, "macro_f1": True, "time": False, "storage": False}


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman test over N folds (blocks) and k configurations. Demšar (2006), Section 3.2.2."""

    n_blocks: int
    n_configs: int
    mean_ranks: pd.Series
    chi2: float
    chi2_p_value: float
    iman_davenport: float
    iman_davenport_p_value: float


@dataclass(frozen=True)
class PostHocResult:
    """Pairwise post-hoc comparison after a significant Friedman test. Demšar (2006), 3.2.2."""

    method: str
    alpha: float
    critical_difference: float
    mean_ranks: pd.Series
    significant: pd.DataFrame  # k x k table, True where two configurations differ


@dataclass(frozen=True)
class ControlComparisonResult:
    """Post-hoc comparison of every configuration against one control."""

    method: str
    control: str
    alpha: float
    critical_difference: float
    mean_ranks: pd.Series
    rank_differences: pd.Series  # control rank minus each rank, > 0 means better than control
    significant: pd.Series  # True where a configuration differs from the control


@dataclass(frozen=True)
class PairedComparisonResult:
    """Two configurations compared fold by fold. Demšar (2006), Section 3.1."""

    first: str
    second: str
    n_blocks: int
    r_plus: float  # rank sum of the folds where `second` is better
    r_minus: float  # rank sum of the folds where `first` is better
    wilcoxon_t: float  # T = min(R+, R-)
    wilcoxon_p_value: float
    t_statistic: float  # paired t-test, reference only; > 0 means `second` is better
    t_p_value: float


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

    Demšar (2006), Section 3.2.2.
    """
    return table.rank(axis=1, ascending=not higher_is_better, method="average")


def mean_ranks(table: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Average rank R_j of each configuration over all folds, best (lowest) first.

    Demšar (2006)
    """
    return rank_table(table, higher_is_better).mean().sort_values()


def friedman_test(table: pd.DataFrame, higher_is_better: bool = True) -> FriedmanResult:
    """Friedman chi-square and the Iman-Davenport F statistic.

    Demšar (2006), Section 3.2.2; Iman and Davenport (1980).
    """
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


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Two-tailed Nemenyi critical value q_alpha, Demšar (2006), Table 5(a).

    The table stops at k = 10. Its caption says the values are the Studentized range
    statistic divided by sqrt(2), so we compute them for any k (infinite degrees of freedom).
    """
    if k < 2:
        raise ValueError("Nemenyi needs at least 2 configurations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return float(sp_stats.studentized_range.ppf(1 - alpha, k, math.inf) / math.sqrt(2))


def critical_difference(k: int, n_blocks: int, q: float) -> float:
    """CD = q * sqrt(k(k+1) / 6N), Demšar (2006), Section 3.2.2."""
    return q * math.sqrt(k * (k + 1) / (6 * n_blocks))


def nemenyi_test(
    table: pd.DataFrame,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> PostHocResult:
    """All-pairs Nemenyi test: two configurations differ if their mean ranks differ by >= CD.

    Demšar (2006), Section 3.2.2. Use it only after a significant Friedman test.
    """
    n_blocks, k = table.shape
    ranks = mean_ranks(table, higher_is_better)
    cd = critical_difference(k, n_blocks, nemenyi_q(k, alpha))

    values = ranks.to_numpy()
    gaps = abs(values[:, None] - values[None, :])
    significant = pd.DataFrame(gaps >= cd, index=ranks.index, columns=ranks.index)

    return PostHocResult(
        method="nemenyi",
        alpha=alpha,
        critical_difference=cd,
        mean_ranks=ranks,
        significant=significant,
    )


def bonferroni_dunn_q(k: int, alpha: float = 0.05) -> float:
    """Two-tailed Bonferroni-Dunn critical value, Demšar (2006), Table 5(b).

    Same CD formula as Nemenyi, but with the normal critical value for alpha / (k - 1),
    because only k - 1 comparisons with the control are made (p. 12).
    """
    if k < 2:
        raise ValueError("Bonferroni-Dunn needs at least 2 configurations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return float(sp_stats.norm.ppf(1 - alpha / (2 * (k - 1))))


def bonferroni_dunn_test(
    table: pd.DataFrame,
    control: str,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> ControlComparisonResult:
    """Compare every configuration with a control: significant if |R_control - R_j| >= CD.

    Demšar (2006), Section 3.2.2. Use it only after a significant Friedman test.
    """
    if control not in table.columns:
        raise ValueError(f"Unknown control: {control!r}")

    n_blocks, k = table.shape
    ranks = mean_ranks(table, higher_is_better)
    cd = critical_difference(k, n_blocks, bonferroni_dunn_q(k, alpha))
    differences = ranks[control] - ranks.drop(control)

    return ControlComparisonResult(
        method="bonferroni-dunn",
        control=control,
        alpha=alpha,
        critical_difference=cd,
        mean_ranks=ranks,
        rank_differences=differences,
        significant=differences.abs() >= cd,
    )


def wilcoxon_signed_ranks(differences: pd.Series) -> tuple[float, float, float]:
    """R+, R- and the exact two-sided p-value of the Wilcoxon signed-ranks test.

    Demšar (2006), Section 3.1.3: rank |d_i| with average ranks for ties; the ranks of
    d_i = 0 are split evenly between R+ and R-, and if their number is odd one is ignored.
    The p-value is exact: under the null hypothesis every non-zero difference is equally
    likely to be positive or negative, so we count all the sign patterns of their ranks.
    """
    # Remove floating-point noise only (e.g. 0.30000000000000004); Demšar (p. 8) warns
    # against rounding to one or two decimals because it creates artificial ties.
    d = differences.astype(float).round(12).reset_index(drop=True)

    zero_positions = d.index[d == 0]
    if len(zero_positions) % 2 == 1:
        d = d.drop(zero_positions[0])

    ranks = d.abs().rank(method="average")
    zero_half = ranks[d == 0].sum() / 2
    r_plus = float(ranks[d > 0].sum() + zero_half)
    r_minus = float(ranks[d < 0].sum() + zero_half)
    t = min(r_plus, r_minus)

    # Exact null distribution of R+. Average ranks are whole or half numbers, so we
    # double them to work with integers: counts[s] = number of sign patterns whose
    # positive ranks add up to s / 2.
    doubled = [round(2 * r) for r in ranks[d != 0]]
    counts = {0: 1}
    for r in doubled:
        new_counts: dict[int, int] = {}
        for total, n in counts.items():
            new_counts[total] = new_counts.get(total, 0) + n  # this rank is negative
            new_counts[total + r] = new_counts.get(total + r, 0) + n  # this rank is positive
        counts = new_counts

    limit = 2 * (t - zero_half)
    lower_tail = sum(n for total, n in counts.items() if total <= limit + 1e-9)
    p_value = min(1.0, 2 * lower_tail / 2 ** len(doubled))
    return r_plus, r_minus, p_value


def compare_two(
    table: pd.DataFrame,
    first: str,
    second: str,
    higher_is_better: bool = True,
) -> PairedComparisonResult:
    """Wilcoxon signed-ranks test (decision) and paired t-test (reference) for two configs.

    Demšar (2006), Section 3.1: Wilcoxon is recommended; the t-test is reported only
    because the assignment mentions it (Sections 1.1 and 1.3).
    """
    for name in (first, second):
        if name not in table.columns:
            raise ValueError(f"Unknown configuration {name!r}")

    sign = 1 if higher_is_better else -1
    differences = sign * (table[second] - table[first])  # > 0 means `second` is better

    r_plus, r_minus, p_value = wilcoxon_signed_ranks(differences)
    t_result = sp_stats.ttest_1samp(differences, 0.0)

    return PairedComparisonResult(
        first=first,
        second=second,
        n_blocks=len(table),
        r_plus=r_plus,
        r_minus=r_minus,
        wilcoxon_t=min(r_plus, r_minus),
        wilcoxon_p_value=p_value,
        t_statistic=float(t_result.statistic),
        t_p_value=float(t_result.pvalue),
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values for a family of m tests (Demšar 2006, Section 3.2.2).

    The i-th smallest p-value is multiplied by (m - i + 1), made non-decreasing and capped
    at 1; a hypothesis is rejected when its adjusted p-value is below alpha.
    """
    ordered = p_values.astype(float).sort_values()
    factors = pd.Series(range(len(ordered), 0, -1), index=ordered.index, dtype=float)
    adjusted = (ordered * factors).cummax().clip(upper=1.0)
    return adjusted.reindex(p_values.index)


def nemenyi_groups(result: PostHocResult) -> list[list[str]]:
    """Maximal groups of consecutive configurations (by mean rank) whose ranks differ by < CD.

    These are the groups joined by a line in the CD diagram, Demšar (2006), Figure 1(a).
    """
    ranks = result.mean_ranks.sort_values()
    names = list(ranks.index)
    values = ranks.to_list()

    groups: list[list[str]] = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and values[j + 1] - values[i] < result.critical_difference:
            j += 1
        if j > i:
            candidate = names[i : j + 1]
            if not any(set(candidate) <= set(group) for group in groups):
                groups.append(candidate)
    return groups


def plot_cd_diagram(
    result: PostHocResult | ControlComparisonResult,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Critical difference diagram; returns the matplotlib Figure.

    Demšar (2006), Section 3.2.4 and Figure 1: mean ranks on an axis with the best (lowest)
    rank on the right. Nemenyi: groups that are not significantly different are joined by a
    thick line (Fig. 1a). Bonferroni-Dunn: the interval of one CD around the control is
    marked (Fig. 1b). The CD is drawn above the axis.
    """
    import matplotlib.pyplot as plt  # only needed here

    ranks = result.mean_ranks.sort_values()
    names = list(ranks.index)
    k = len(names)
    cd = result.critical_difference

    control = result.control if isinstance(result, ControlComparisonResult) else None
    if isinstance(result, ControlComparisonResult):
        centre = ranks[result.control]
        bars = [(max(1.0, centre - cd), min(float(k), centre + cd))]
    else:
        bars = [(ranks[group[0]], ranks[group[-1]]) for group in nemenyi_groups(result)]

    half = (k + 1) // 2
    right = names[:half]  # best half, labelled on the right
    left = names[half:][::-1]  # worst half, labelled on the left (worst at the top)
    step = 0.35
    first_row = -0.25 - 0.15 * len(bars)
    n_rows = max(len(right), len(left))

    fig, ax = plt.subplots(figsize=(8, 1.5 + step * n_rows + 0.15 * len(bars)))
    ax.set_xlim(k + 0.5, 0.5)  # inverted: rank 1 (best) on the right
    ax.set_ylim(first_row - step * n_rows, 0.9)
    ax.axis("off")

    # Rank axis with integer ticks.
    ax.hlines(0, 1, k, color="black")
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.08, color="black")
        ax.text(r, 0.12, str(r), ha="center", va="bottom")

    # Critical difference above the axis.
    ax.hlines(0.6, 1, min(1 + cd, k), color="black", linewidth=2)
    ax.text(1 + min(cd, k - 1) / 2, 0.65, f"CD = {cd:.2f}", ha="center", va="bottom")

    # Groups that are not significantly different (Nemenyi) or the control interval.
    pad = 0.0 if isinstance(result, ControlComparisonResult) else 0.03
    for i, (start, end) in enumerate(bars):
        y = -0.15 * (i + 1)
        ax.hlines(y, start - pad, end + pad, color="black", linewidth=4)

    # One label per configuration: best half on the right, worst half on the left.
    for row, name in enumerate(right):
        y = first_row - step * row
        ax.vlines(ranks[name], y, 0, color="gray", linewidth=0.8)
        ax.hlines(y, 0.9, ranks[name], color="gray", linewidth=0.8)
        ax.text(
            0.85,
            y,
            f"{name} ({ranks[name]:.2f})",
            ha="left",
            va="center",
            fontweight="bold" if name == control else "normal",
        )
    for row, name in enumerate(left):
        y = first_row - step * row
        ax.vlines(ranks[name], y, 0, color="gray", linewidth=0.8)
        ax.hlines(y, ranks[name], k + 0.1, color="gray", linewidth=0.8)
        ax.text(
            k + 0.15,
            y,
            f"{name} ({ranks[name]:.2f})",
            ha="right",
            va="center",
            fontweight="bold" if name == control else "normal",
        )
    if title:
        ax.set_title(title)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig
