"""Exploratory data analysis of the preprocessed datasets.

Saves a few figures (and optionally the preprocessed fold as CSV), prints a summary and writes
``report.md`` with every dataset figure quoted in the report (Section 2): ambiguity codes,
mutual information per position, splice motifs, duplicates, vowel speakers, test values
clipped by the scaling and the 1-NN accuracy.
Run from the project root:  python experiments/eda.py --dataset all --fold 0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from arff_parser import load_fold  # noqa: E402
from models.knn.distances import euclidean  # noqa: E402
from models.knn.knn_utils import kNNAlgorithm  # noqa: E402
from models.knn.voting import majority  # noqa: E402
from preprocessing import Preprocessed, Preprocessor, preprocess_fold  # noqa: E402

DPI = 150
NUCLEOTIDES = ("A", "C", "G", "T")
# Positions around the splice junction (1-based, between 30 and 31) quoted in the report.
JUNCTION = (28, 35)
VOWEL_NOMINAL = ("Train_or_Test", "Speaker_Number", "Sex")


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


# ---------------------------------------------------------------------------
# report.md: dataset checks quoted in the report
# ---------------------------------------------------------------------------
def pct(part: float, total: float, digits: int = 1) -> str:
    return f"{100 * part / total:.{digits}f}%"


def md_table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines + [""]


def all_folds(dataset: str) -> list[tuple]:
    """Raw (train, test) ARFF data of the 10 folds."""
    return [load_fold(dataset, f) for f in range(config.N_FOLDS)]


def whole_dataset(folds: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Every instance once: the union of the 10 test parts (raw values and labels)."""
    X = np.vstack([test.X for _, test in folds])
    y = np.concatenate([test.y for _, test in folds])
    return X, y


def class_counts_section(dataset: str, folds: list[tuple]) -> list[str]:
    _, y = whole_dataset(folds)
    counts = Counter(y)
    n_train = [train.n_instances for train, _ in folds]
    n_test = [test.n_instances for _, test in folds]
    rows = [[c, n, pct(n, len(y))] for c, n in sorted(counts.items())]
    return [
        f"Instances: {len(y)}; train per fold {min(n_train)}-{max(n_train)}, "
        f"test per fold {min(n_test)}-{max(n_test)}.",
        "",
        *md_table(["class", "instances", "share"], rows),
    ]


def mutual_information(codes: np.ndarray, y_codes: np.ndarray) -> np.ndarray:
    """Mutual information (bits) between each nominal column and the class."""
    mi = np.zeros(codes.shape[1])
    for j in range(codes.shape[1]):
        joint = Counter(zip(codes[:, j], y_codes, strict=True))
        px, py, n = Counter(codes[:, j]), Counter(y_codes), len(y_codes)
        mi[j] = sum(c / n * np.log2(c * n / (px[a] * py[b])) for (a, b), c in joint.items())
    return mi


def splice_positions_section(train: Preprocessed, fold: int) -> list[str]:
    """Mutual information, splice motifs and the per-class nucleotide shares of Figure 1."""
    lo, hi = JUNCTION
    mi = mutual_information(train.X_nom, train.y_codes)
    total = mi.sum()
    top10 = np.sort(mi)[-10:].sum()
    junction = mi[lo - 1 : hi].sum()
    outside = sorted(
        ((j + 1, mi[j]) for j in range(len(mi)) if not lo <= j + 1 <= hi), key=lambda t: -t[1]
    )
    lines = [
        f"### Information per position (training part of fold {fold})",
        "",
        f"- Positions {lo}-{hi} carry {pct(junction, total)} of the total mutual "
        f"information ({total:.3f} bits); outside them: {pct(total - junction, total)}.",
        f"- The ten most informative positions carry {pct(top10, total)}.",
        f"- Positions with less than 0.01 bits: {int(np.sum(mi < 0.01))} of {len(mi)}.",
        "- Most informative positions outside the junction: "
        + ", ".join(f"{p} ({v:.3f} bits)" for p, v in outside[:6])
        + ".",
        "",
    ]

    # Motifs: G at 31 and T at 32 (EI, donor), A at 29 and G at 30 (IE, acceptor).
    def share_with(class_label: str, motif: dict[int, str]) -> tuple[int, int]:
        rows = train.X_nom[train.y_codes == train.class_values.index(class_label)]
        hit = np.ones(len(rows), dtype=bool)
        for pos, letter in motif.items():
            hit &= rows[:, pos - 1] == train.nom_values[pos - 1].index(letter)
        return int(hit.sum()), len(rows)

    gt = {31: "G", 32: "T"}
    ag = {29: "A", 30: "G"}
    rows = []
    for label in train.class_values:
        (g, n), (a, _) = share_with(label, gt), share_with(label, ag)
        rows.append([label, n, f"{g} ({pct(g, n)})", f"{a} ({pct(a, n)})"])
    lines += [f"### Splice motifs (training part of fold {fold})", ""]
    lines += md_table(["class", "rows", "GT at 31-32", "AG at 29-30"], rows)

    # Figure 1 caption: share of C+T per region and class, and how far N is from uniform.
    regions = [(1, 20), (21, lo - 1), (lo, hi), (hi + 1, len(mi))]
    rows = []
    for code, label in enumerate(train.class_values):
        freq = nucleotide_frequencies(train, code)  # (4, positions)
        pyrimidines = freq[NUCLEOTIDES.index("C")] + freq[NUCLEOTIDES.index("T")]
        cells = [f"{pyrimidines[a - 1 : b].mean():.3f}" for a, b in regions]
        rows.append([label, *cells, f"{freq.min():.3f}-{freq.max():.3f}"])
    lines += [
        f"### Nucleotide shares per class (Figure 1, training part of fold {fold})",
        "",
        "Mean share of C+T (pyrimidines) in each region, and range of the A/C/G/T shares over "
        "all positions (a uniform class would be 0.25 everywhere).",
        "",
    ]
    lines += md_table(["class", *[f"C+T {a}-{b}" for a, b in regions], "A/C/G/T share range"], rows)
    return lines


def splice_values_section(folds: list[tuple]) -> list[str]:
    X, _ = whole_dataset(folds)
    values = Counter(X.ravel())
    ambiguous = {v: c for v, c in values.items() if v not in NUCLEOTIDES}
    return [
        "### Values",
        "",
        f"- Cells: {X.size} ({X.shape[0]} rows x {X.shape[1]} positions).",
        f"- Ambiguity codes (not A/C/G/T): {sum(ambiguous.values())} cells "
        f"({pct(sum(ambiguous.values()), X.size, 3)}): "
        + ", ".join(f"{v} {c}" for v, c in sorted(ambiguous.items()))
        + ".",
        "",
    ]


def splice_duplicates_section(folds: list[tuple]) -> list[str]:
    """Identical sequences in the whole dataset, inside each training fold and across train/test."""
    X, y = whole_dataset(folds)
    keys = [tuple(row) for row in X]
    copies = Counter(keys)
    labels: dict[tuple, set] = {}
    for key, label in zip(keys, y, strict=True):
        labels.setdefault(key, set()).add(label)
    multi_class = sum(len(v) > 1 for v in labels.values())

    rows, with_copy, same_class, n_test = [], 0, 0, 0
    for f, (train, test) in enumerate(folds):
        train_keys = [tuple(row) for row in train.X]
        train_labels: dict[tuple, set] = {}
        for key, label in zip(train_keys, train.y, strict=True):
            train_labels.setdefault(key, set()).add(label)
        extra = len(train_keys) - len(train_labels)
        hits = [
            (tuple(row), label)
            for row, label in zip(test.X, test.y, strict=True)
            if tuple(row) in train_labels
        ]
        with_copy += len(hits)
        same_class += sum(train_labels[key] == {label} for key, label in hits)
        n_test += test.n_instances
        rows.append([f, extra, pct(extra, len(train_keys)), len(hits)])

    return [
        "### Duplicates",
        "",
        f"- {len(keys)} rows, {len(copies)} distinct sequences, "
        f"{len(keys) - len(copies)} extra copies.",
        f"- Sequences that appear with more than one class: {multi_class}.",
        f"- Test rows with an identical row in their own training fold: {with_copy} of {n_test} "
        f"({pct(with_copy, n_test)}); with the same (and only) class in train: {same_class}.",
        "",
        *md_table(
            ["fold", "extra copies in train", "share of train", "test rows with a copy"], rows
        ),
    ]


def vowel_speakers_section(folds: list[tuple]) -> list[str]:
    """Speakers, (speaker, vowel) overlap between train and test, and nearest neighbours."""
    first_train = folds[0][0]
    idx = {name: first_train.attribute_names.index(name) for name in VOWEL_NOMINAL}
    X, y = whole_dataset(folds)
    sex = dict(zip(X[:, idx["Speaker_Number"]], X[:, idx["Sex"]], strict=True))
    identical = len(X) - len({(*row, label) for row, label in zip(map(tuple, X), y, strict=True)})

    pairs_covered = speakers_in_all = True
    same_speaker = same_both = total = 0
    numeric_only = Preprocessor(drop=VOWEL_NOMINAL)
    for train_raw, test_raw in folds:
        s_tr, s_te = train_raw.X[:, idx["Speaker_Number"]], test_raw.X[:, idx["Speaker_Number"]]
        train_pairs = set(zip(s_tr, train_raw.y, strict=True))
        pairs_covered &= all(p in train_pairs for p in zip(s_te, test_raw.y, strict=True))
        speakers_in_all &= len(set(s_te)) == len(sex)

        # Nearest training case with the 10 numeric features only (Euclidean, first index on ties)
        train = numeric_only.fit_transform(train_raw)
        test = numeric_only.transform(test_raw)
        nearest = np.argmin(euclidean(train, test), axis=1)
        speaker_hit = s_tr[nearest] == s_te
        same_speaker += int(speaker_hit.sum())
        same_both += int((speaker_hit & (train_raw.y[nearest] == test_raw.y)).sum())
        total += len(s_te)

    n_sex = Counter(sex.values())
    return [
        "### Speakers",
        "",
        f"- Speakers: {len(sex)} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(n_sex.items()))
        + ").",
        f"- Identical rows (all columns and class): {identical}.",
        "- Every (speaker, vowel) pair of a test fold is also in its training fold: "
        f"{pairs_covered}.",
        f"- All {len(sex)} speakers appear in every test fold: {speakers_in_all}.",
        f"- Nearest training case (10 numeric features, Euclidean) of the same speaker: "
        f"{same_speaker} of {total} ({pct(same_speaker, total)}); same speaker and vowel: "
        f"{same_both} ({pct(same_both, total)}).",
        "",
    ]


def clipping_section(dataset: str, folds: list[tuple]) -> list[str]:
    """Numeric test cells outside the training range of their fold (clipped to 0 or 1)."""
    clipped = cells = 0
    for train, test in folds:
        tr, te = train.numeric_matrix(), test.numeric_matrix()
        if tr.size == 0:
            return []
        lo, hi = np.nanmin(tr, axis=0), np.nanmax(tr, axis=0)
        clipped += int(np.sum((te < lo) | (te > hi)))
        cells += te.size
    return [
        "### Clipping",
        "",
        f"- Numeric test cells outside the training range of their fold: {clipped} of {cells} "
        f"({pct(clipped, cells, 2)}).",
        "",
    ]


def one_nn_section(dataset: str) -> list[str]:
    """1-NN (Euclidean, majority vote) over the 10 folds with the default preprocessing."""
    accs = []
    for f in range(config.N_FOLDS):
        train, test, _ = preprocess_fold(dataset, f)
        pred = np.asarray(kNNAlgorithm(train, test, 1, euclidean, majority))
        accs.append(float(np.mean(pred == test.y_codes)))
    return [
        "### 1-NN",
        "",
        f"- 1-NN (Euclidean, majority vote): mean accuracy {np.mean(accs):.4f} "
        f"(min {min(accs):.4f}, max {max(accs):.4f}) over the 10 folds.",
        "",
    ]


def dataset_report(dataset: str, fold: int, train: Preprocessed) -> list[str]:
    folds = all_folds(dataset)
    lines = [f"## {dataset}", ""]
    lines += class_counts_section(dataset, folds)
    lines += [f"One-hot width (SVM and RF input): {train.to_onehot().shape[1]}.", ""]
    if dataset == "splice":
        lines += splice_values_section(folds)
        lines += splice_positions_section(train, fold)
        lines += splice_duplicates_section(folds)
    if dataset == "vowel":
        lines += vowel_speakers_section(folds)
    lines += clipping_section(dataset, folds)
    lines += one_nn_section(dataset)
    return lines


def write_report(lines: list[str], path: Path) -> None:
    # Exactly one newline at the end (the tables add a trailing blank line).
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"  saved {path}")


def run(dataset: str, fold: int, out: Path, csv: bool) -> list[str]:
    """Figures, CSVs and summary of one dataset; returns its section of report.md."""
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
    return dataset_report(dataset, fold, train)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EDA of the preprocessed datasets.")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument("--fold", type=int, choices=range(config.N_FOLDS), default=0)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eda")
    parser.add_argument("--csv", action="store_true", help="also save the fold as CSV (one-hot)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    datasets = config.DATASETS if args.dataset == "all" else (args.dataset,)
    lines = [
        "# EDA report",
        "",
        "Generated by `experiments/eda.py`. Dataset-wide figures use the 10 folds (each instance "
        "is in exactly one test part); per-position figures use the training part of fold "
        f"{args.fold}.",
        "",
    ]
    for dataset in datasets:
        lines += run(dataset, args.fold, args.out, args.csv)
    write_report(lines, args.out / "report.md")


if __name__ == "__main__":
    main()
