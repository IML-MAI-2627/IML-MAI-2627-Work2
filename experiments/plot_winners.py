"""Retrain the winners of grid_search_v2 and plot ROC, PR, confusion matrices and trade-offs.

Reads ``winners.json`` written by ``grid_search_v2.py --stage evaluate``. Run from the
project root:

    python experiments/plot_winners.py --results results/gridsearch_v2

For every dataset and winner (best kNN / SVM / RF, without and with the best reduction
technique) the 10 folds are run again with per-class scores switched on. Seeds are fixed, so
the predictions are the ones of the grid search: the script checks that the mean accuracy
matches winners.json. Scores are cached in ``<results>/plots/cache`` (one .npz per winner),
so the plots can be redrawn without retraining (``--retrain`` forces it).

Curves pool the out-of-fold scores of the 10 test folds (each instance is scored once).
Multiclass curves are one-vs-rest: ROC is macro-averaged over the classes, PR is
micro-averaged; the legend gives the macro AUC / macro average precision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from grid_search_v2 import (  # noqa: E402
    CLASSIFIERS,
    KNN_BASE_KEYS,
    load_dataset,
    matches,
    parse_params,
)

import config  # noqa: E402
from experiment_runner import confusion_matrix, load_fold  # noqa: E402

# Fixed categorical order (identity follows the model, never its rank). Reduced = dashed.
COLORS = {"knn": "#2a78d6", "svm": "#eb6834", "rf": "#1baf7a"}
NAMES = {"knn": "kNN", "svm": "SVM", "rf": "RF"}
MARKERS = {"none": "o", "fcnn": "s", "eenth": "^", "drop": "D", "del": "v"}
WINNER_KEYS = ("knn", "svm", "rf", "knn_reduced", "svm_reduced", "rf_reduced")

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984",
        "font.size": 10,
    }
)


def run_winner(dataset: str, winner: dict) -> dict[str, np.ndarray]:
    """Run one winner on the 10 folds, like run_experiment, keeping the scores."""
    cls, _ = CLASSIFIERS[winner["model"]]
    model = cls(dict(winner["params"]))
    model.return_scores = True
    config.set_seed()  # same seed reset as run_experiment
    y_true, y_pred, scores, folds = [], [], [], []
    for fold in range(config.N_FOLDS):
        train, test, _ = load_fold(dataset, fold)
        result = model.fit_predict(train, test)
        y_true.append(np.asarray(test.y_codes))
        y_pred.append(np.asarray(result.predictions, dtype=np.int64))
        scores.append(result.scores)
        folds.append(np.full(len(test.y_codes), fold))
    return {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "scores": np.vstack(scores),
        "fold": np.concatenate(folds),
    }


def load_or_run(cache: Path, dataset: str, key: str, winner: dict, retrain: bool) -> dict:
    path = cache / f"{dataset}_{key}.npz"
    if path.exists() and not retrain:
        return dict(np.load(path))
    data = run_winner(dataset, winner)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    # Same seeds and folds as the grid search: the accuracy must be the one it reported.
    per_fold = [
        np.mean(data["y_true"][data["fold"] == f] == data["y_pred"][data["fold"] == f])
        for f in range(config.N_FOLDS)
    ]
    if not np.isclose(np.mean(per_fold), winner["accuracy"]):
        print(
            f"  [!] {dataset} {key}: accuracy {np.mean(per_fold):.4f} differs from the "
            f"grid search ({winner['accuracy']:.4f})"
        )
    return data


def onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=int)[y]


# ROC / PR with numpy (own code): scikit-learn is only used for SVM, RF and preprocessing.
def _cumulative_counts(positives: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True and false positives when the threshold goes down through each distinct score."""
    order = np.argsort(-score, kind="mergesort")
    positives, score = positives[order], score[order]
    last_of_each_score = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tps = np.cumsum(positives)[last_of_each_score]
    fps = last_of_each_score + 1 - tps
    return tps.astype(float), fps.astype(float)


def roc_curve(positives: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate) from (0, 0) to (1, 1); ties form one step."""
    tps, fps = _cumulative_counts(np.asarray(positives, dtype=bool), np.asarray(score, dtype=float))
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def precision_recall(positives: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(precision, recall) for decreasing thresholds, starting at recall 0 with precision 1."""
    tps, fps = _cumulative_counts(np.asarray(positives, dtype=bool), np.asarray(score, dtype=float))
    return np.r_[1.0, tps / (tps + fps)], np.r_[0.0, tps / tps[-1]]


def average_precision(positives: np.ndarray, score: np.ndarray) -> float:
    """AP = sum over thresholds of (R_n - R_{n-1}) * P_n (no interpolation)."""
    precision, recall = precision_recall(positives, score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def macro_roc(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Macro-averaged one-vs-rest ROC: mean TPR over classes on a common FPR grid."""
    grid = np.linspace(0, 1, 201)
    tprs, aucs = [], []
    for c in range(scores.shape[1]):
        positives = y == c
        if positives.all() or not positives.any():
            continue
        fpr, tpr = roc_curve(positives, scores[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(float(np.trapezoid(tpr, fpr)))
    curve = np.mean(tprs, axis=0)
    curve[0] = 0.0
    return grid, curve, float(np.mean(aucs))


def style(key: str) -> dict:
    base = key.replace("_reduced", "")
    reduced = key.endswith("_reduced")
    return {
        "color": COLORS[base],
        "linestyle": "--" if reduced else "-",
        "linewidth": 2,
    }


def label(key: str, winners: dict) -> str:
    base = key.replace("_reduced", "")
    if key.endswith("_reduced"):
        return f"{NAMES[base]} + {winners.get('best_reduction', 'reduction')}"
    return NAMES[base]


def plot_roc_pr(dataset: str, runs: dict, winners: dict, n_classes: int, out: Path) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for key, data in runs.items():
        y, s = data["y_true"], data["scores"]
        fpr, tpr, roc_auc = macro_roc(y, s)
        ax_roc.plot(fpr, tpr, label=f"{label(key, winners)} (AUC {roc_auc:.3f})", **style(key))
        y1 = onehot(y, n_classes)
        precision, recall = precision_recall(y1.ravel(), s.ravel())
        ap = np.mean(
            [average_precision(y1[:, c], s[:, c]) for c in range(n_classes) if y1[:, c].any()]
        )
        ax_pr.plot(recall, precision, label=f"{label(key, winners)} (AP {ap:.3f})", **style(key))
    ax_roc.plot([0, 1], [0, 1], color="#b5b4ae", linewidth=1, linestyle=":")
    ax_roc.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{dataset}: ROC (one-vs-rest, macro)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{dataset}: precision-recall (micro)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"roc_pr_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_confusions(dataset: str, runs: dict, winners: dict, classes, out: Path) -> None:
    keys = list(runs)
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    size = 3.2 if len(classes) <= 4 else 4.2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size), squeeze=False)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    for ax, key in zip(axes.ravel(), keys, strict=False):
        data = runs[key]
        matrix = confusion_matrix(data["y_true"], data["y_pred"], len(classes))
        rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(classes)), classes, rotation=90 if len(classes) > 4 else 0)
        ax.set_yticks(range(len(classes)), classes)
        ax.set(xlabel="Predicted", ylabel="True", title=label(key, winners))
        if len(classes) <= 4:  # counts are readable only for a few classes
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ink = "white" if rates[i, j] > 0.6 else "#0b0b0b"
                    ax.text(j, i, matrix[i, j], ha="center", va="center", color=ink, fontsize=9)
    fig.suptitle(f"{dataset}: confusion matrices (10 test folds, row-normalised colour)")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(dataset: str, results: Path, winners: dict, out: Path) -> None:
    """Accuracy vs storage (left) and vs solve time (right) for each model and technique."""
    path = results / dataset / "summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path, keep_default_na=False)
    summary["p"] = summary["params"].map(parse_params)
    fig, (ax_st, ax_t) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for base in ("knn", "svm", "rf"):
        if base not in winners:
            continue
        params = {k: str(v) for k, v in winners[base]["params"].items()}
        keys = KNN_BASE_KEYS if base == "knn" else tuple(k for k in params if k != "reduction")
        rows = summary[
            (summary["key"] == base)
            & summary["p"].map(lambda p, b=params, k=keys: matches(p, b, k))
        ]
        if base == "knn":  # the winner's variant of each technique only
            champions = winners.get("reduction_champions", {})
            rows = rows[
                rows["p"].map(
                    lambda p, ch=champions: (
                        p.get("reduction", "none") == "none"
                        or all(p.get(k) == str(v) for k, v in ch.get(p["reduction"], {}).items())
                    )
                )
            ]
        for _, row in rows.iterrows():
            marker = MARKERS.get(row["reduction"], "o")
            for ax, x in ((ax_st, row["storage"]), (ax_t, row["solve_time"])):
                ax.scatter(
                    x,
                    row["accuracy"],
                    s=64,
                    marker=marker,
                    color=COLORS[base],
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=3,
                )
    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=8, label=NAMES[m])
        for m, c in COLORS.items()
    ] + [
        plt.Line2D([], [], color="#52514e", marker=mk, linestyle="", markersize=8, label=t)
        for t, mk in MARKERS.items()
    ]
    ax_st.set(xlabel="Storage (% of the training set)", ylabel="Mean accuracy")
    ax_t.set(xlabel="Mean solve time per fold (s, without reduction)", xscale="log")
    ax_t.legend(handles=handles, fontsize=8, frameon=False, loc="lower right", ncol=2)
    fig.suptitle(f"{dataset}: accuracy vs storage and time (best models, each technique)")
    fig.tight_layout()
    fig.savefig(out / f"tradeoff_{dataset}.png", dpi=150)
    plt.close(fig)


TECHNIQUES = ("none", "fcnn", "eenth", "drop", "del")


def reduction_table(results: Path, dataset: str, winners: dict) -> pd.DataFrame:
    """Per fold rows of the best kNN / SVM / RF without reduction and with each technique
    (the variant chosen on the best kNN), labelled with ``model`` and ``technique``."""
    df = load_dataset(results, dataset)
    champions = winners.get("reduction_champions", {})

    def chosen_variant(p: dict) -> bool:
        technique = p.get("reduction", "none")
        if technique == "none":
            return True
        if technique not in champions:
            return False
        return all(p.get(k) == str(v) for k, v in champions[technique].items())

    parts = []
    for base in ("knn", "svm", "rf"):
        if base not in winners:
            continue
        params = {k: str(v) for k, v in winners[base]["params"].items()}
        keys = KNN_BASE_KEYS if base == "knn" else tuple(k for k in params if k != "reduction")
        rows = df[
            (df["key"] == base)
            & df["p"].map(lambda p, b=params, k=keys: matches(p, b, k))
            & df["p"].map(chosen_variant)
        ]
        parts.append(rows.assign(model=base, technique=rows["reduction"]))
    return pd.concat(parts, ignore_index=True)


def plot_reduction(dataset: str, results: Path, winners: dict, out: Path) -> None:
    """Accuracy, solve time, reduction time and storage per technique and model (#19, #20)."""
    table = reduction_table(results, dataset, winners)
    stats_ = table.groupby(["model", "technique"]).agg(
        acc=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        solve=("solve_time", "mean"),
        reduce=("reduction_time", "mean"),
        storage=("storage", "mean"),
    )
    techniques = [t for t in TECHNIQUES if t in set(table["technique"])]
    models = [m for m in ("knn", "svm", "rf") if m in set(table["model"])]
    x = np.arange(len(techniques))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (ax_acc, ax_solve), (ax_red, ax_sto) = axes

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        rows = stats_.loc[model].reindex(techniques)
        bar = {"width": width, "color": COLORS[model], "edgecolor": "white", "linewidth": 1}
        ax_acc.bar(
            x + offset,
            rows["acc"],
            yerr=rows["acc_std"],
            label=NAMES[model],
            error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2},
            **bar,
        )
        ax_solve.bar(x + offset, rows["solve"], **bar)
        ax_sto.bar(x + offset, rows["storage"], **bar)

    # The reduced set is shared by the three models, so its cost is one bar per technique.
    reduced = [t for t in techniques if t != "none"]
    cost = stats_.loc["knn"].reindex(reduced)["reduce"] if "knn" in models else None
    if cost is not None:
        ax_red.bar(np.arange(len(reduced)), cost, width=0.5, color="#8a8984")
        ax_red.set_xticks(np.arange(len(reduced)), reduced)
        for xi, value in enumerate(cost):
            ax_red.text(xi, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=8)

    low = max(0.0, float((stats_["acc"] - stats_["acc_std"]).min()) - 0.02)
    ax_acc.set(ylabel="Mean accuracy (± std over folds)", ylim=(low, 1.0), title="Accuracy")
    ax_solve.set(
        ylabel="Seconds per fold (log)",
        yscale="log",
        title="Solve time (fit + predict, without the reduction)",
    )
    ax_red.set(ylabel="Seconds per fold", title="Reduction time (shared by the three models)")
    ax_sto.axhline(100, color="#b5b4ae", linewidth=1, linestyle=":")
    ax_sto.set(ylabel="% of the training set", title="Storage (kNN includes its retention policy)")
    for ax in (ax_acc, ax_solve, ax_sto):
        ax.set_xticks(x, techniques)
        ax.grid(axis="x", visible=False)
    ax_red.grid(axis="x", visible=False)
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.95)
    )
    fig.suptitle(f"{dataset}: effect of instance reduction on the best kNN, SVM and RF")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"reduction_{dataset}.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plots of the grid_search_v2 winners.")
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR / "gridsearch_v2")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument("--retrain", action="store_true", help="ignore the cached scores")
    args = parser.parse_args(argv)

    all_winners = json.loads((args.results / "winners.json").read_text())
    out = args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    datasets = list(all_winners) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        winners = all_winners[dataset]
        _, test, _ = load_fold(dataset, 0)
        classes = list(test.class_values)
        runs = {}
        for key in WINNER_KEYS:
            if key in winners:
                print(f"[{dataset}] {key}")
                runs[key] = load_or_run(out / "cache", dataset, key, winners[key], args.retrain)
        plot_roc_pr(dataset, runs, winners, len(classes), out)
        plot_confusions(dataset, runs, winners, classes, out)
        plot_tradeoff(dataset, args.results, winners, out)
        plot_reduction(dataset, args.results, winners, out)
        print(f"[{dataset}] plots saved to {out}")


if __name__ == "__main__":
    main()
