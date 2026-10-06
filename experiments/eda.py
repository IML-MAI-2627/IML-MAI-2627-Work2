"""Exploratory data analysis of the preprocessed datasets.

Saves a few figures (and optionally the preprocessed fold as CSV) and prints a summary.
Run from the project root:  python experiments/eda.py --dataset all --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, ssh)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make src/ importable when the script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from preprocessing import Preprocessed, preprocess_fold  # noqa: E402

DPI = 150
NUCLEOTIDES = ("A", "C", "G", "T")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


def class_counts(train: Preprocessed, test: Preprocessed) -> pd.Series:
    """Instances per class in the whole dataset (train + test of the fold)."""
    codes = np.concatenate([train.y_codes, test.y_codes])
    counts = np.bincount(codes, minlength=len(train.class_values))
    return pd.Series(counts, index=train.class_values, name="count")


def print_summary(dataset: str, fold: int, train: Preprocessed, test: Preprocessed) -> None:
    counts = class_counts(train, test)
    share = counts.max() / counts.sum()
    print(f"\n== {dataset} (fold {fold}) ==")
    print(f"  train: X_num {train.X_num.shape}, X_nom {train.X_nom.shape}")
    print(f"  test:  X_num {test.X_num.shape}, X_nom {test.X_nom.shape}")
    print(f"  numeric attributes: {len(train.num_names)}")
    print(f"  nominal attributes: {len(train.nom_names)}")
    print(f"  one-hot width: {train.to_onehot().shape[1]}")
    print(f"  classes: {len(counts)}, instances: {counts.sum()}")
    print(f"  class counts: {counts.to_dict()}")
    print(f"  majority class: {counts.idxmax()} ({share:.1%})")


def plot_classes(dataset: str, train: Preprocessed, test: Preprocessed, path: Path) -> None:
    counts = class_counts(train, test)
    share = counts.max() / counts.sum()
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(counts) + 2), 3.8))
    sns.barplot(x=counts.index, y=counts.values, color="steelblue", ax=ax)
    ax.bar_label(ax.containers[0], fontsize=8)
    ax.set_xlabel("class")
    ax.set_ylabel("instances")
    ax.set_title(f"{dataset}: class distribution\nmajority class {counts.idxmax()} = {share:.1%}")
    _save(fig, path)


def plot_numeric_histograms(dataset: str, train: Preprocessed, path: Path) -> None:
    n = len(train.num_names)
    n_cols = 5
    n_rows = -(-n // n_cols)  # round up
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.6 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis("off")
            continue
        sns.histplot(train.X_num[:, k], bins=20, binrange=(0, 1), ax=ax)
        ax.set_title(train.num_names[k], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
    fig.suptitle(f"{dataset}: numeric attributes after min-max (train)")
    _save(fig, path)


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First two principal components with numpy SVD, and the share of variance of each."""
    centered = X - X.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular**2 / np.sum(singular**2)
    return centered @ vt[:2].T, explained[:2]


def plot_pca(dataset: str, train: Preprocessed, path: Path) -> None:
    points, explained = pca_2d(train.X_num)
    colors = plt.get_cmap("tab20").colors  # 20 distinct colours, enough for 11 classes
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for code, label in enumerate(train.class_values):
        rows = train.y_codes == code
        ax.scatter(points[rows, 0], points[rows, 1], s=12, color=colors[code], label=label)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title(f"{dataset}: PCA of the numeric attributes (train)")
    ax.legend(title="class", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, path)


def nucleotide_frequencies(train: Preprocessed, class_code: int) -> np.ndarray:
    """Share of A/C/G/T at each position, for the rows of one class. Shape (4, positions)."""
    rows = train.X_nom[train.y_codes == class_code]
    freq = np.zeros((len(NUCLEOTIDES), rows.shape[1]))
    for j, values in enumerate(train.nom_values):
        for i, letter in enumerate(NUCLEOTIDES):
            if letter in values:
                freq[i, j] = np.mean(rows[:, j] == values.index(letter))
    return freq


def plot_nucleotide_positions(dataset: str, train: Preprocessed, path: Path) -> None:
    freqs = [nucleotide_frequencies(train, c) for c in range(len(train.class_values))]
    vmax = max(f.max() for f in freqs)  # same colour scale in every panel
    n_pos = freqs[0].shape[1]
    ticks = np.arange(0, n_pos, 5)

    fig, axes = plt.subplots(len(freqs), 1, figsize=(13, 2.2 * len(freqs)), sharex=True)
    for ax, freq, label in zip(axes, freqs, train.class_values, strict=True):
        sns.heatmap(freq, ax=ax, cmap="viridis", vmin=0, vmax=vmax, cbar_kws={"label": "share"})
        ax.set_yticks(np.arange(len(NUCLEOTIDES)) + 0.5, NUCLEOTIDES, rotation=0)
        ax.set_xticks(ticks + 0.5, ticks + 1, rotation=0)
        ax.set_title(f"class {label}", fontsize=10)
    axes[-1].set_xlabel("position (attribute number)")
    fig.suptitle(
        f"{dataset}: nucleotide frequency per position and class (train; "
        "rare ambiguous codes D/N/S/R not shown)"
    )
    _save(fig, path)


def write_csv(data: Preprocessed, path: Path) -> None:
    """One-hot version of a fold plus a final 'class' column with the original label."""
    onehot = data.to_onehot()
    names = list(data.onehot_names())
    n_num = len(data.num_names)
    frame = pd.concat(
        [
            pd.DataFrame(onehot[:, :n_num], columns=names[:n_num]),
            pd.DataFrame(onehot[:, n_num:].astype(int), columns=names[n_num:]),  # 0/1 as int
            pd.Series(data.y, name="class"),
        ],
        axis=1,
    )
    frame.to_csv(path, index=False)
    print(f"  saved {path}")


def run(dataset: str, fold: int, out: Path, csv: bool) -> None:
    train, test, _ = preprocess_fold(dataset, fold)
    prefix = f"{dataset}_fold{fold}"

    print_summary(dataset, fold, train, test)
    plot_classes(dataset, train, test, out / f"{prefix}_classes.png")
    if dataset == "vowel":
        plot_numeric_histograms(dataset, train, out / f"{prefix}_hist.png")
        plot_pca(dataset, train, out / f"{prefix}_pca.png")
    if dataset == "splice":
        plot_nucleotide_positions(dataset, train, out / f"{prefix}_positions.png")
    if csv:
        write_csv(train, out / f"{prefix}_train.csv")
        write_csv(test, out / f"{prefix}_test.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EDA of the preprocessed datasets.")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument("--fold", type=int, choices=range(config.N_FOLDS), default=0)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eda")
    parser.add_argument("--csv", action="store_true", help="also save the fold as CSV (one-hot)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    datasets = config.DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run(dataset, args.fold, args.out, args.csv)


if __name__ == "__main__":
    main()
