"""Retrain the winners of grid_search and plot ROC, PR, confusion matrices and trade-offs.

Reads ``winners.json`` written by ``grid_search.py --stage evaluate``. Run from the
project root:

    python experiments/plot_winners.py --results results/gridsearch

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

from grid_search import (  # noqa: E402
    CLASSIFIERS,
    KNN_BASE_KEYS,
    load_dataset,
    matches,
    md_table,
    parse_params,
)

import config  # noqa: E402
import stats  # noqa: E402
from experiment_runner import confusion_matrix, load_fold  # noqa: E402

# Fixed categorical order (identity follows the model, never its rank). Reduced = dashed.
COLORS = {"knn": "#2a78d6", "svm": "#eb6834", "rf": "#1baf7a"}
NAMES = {"knn": "kNN", "svm": "SVM", "rf": "RF"}
MARKERS = {"none": "o", "fcnn": "s", "ennth": "^", "drop": "D", "del": "v"}
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
    fit_times, predict_times, reduction_times = [], [], []
    for fold in range(config.N_FOLDS):
        train, test, _ = load_fold(dataset, fold)
        result = model.fit_predict(train, test)
        y_true.append(np.asarray(test.y_codes))
        y_pred.append(np.asarray(result.predictions, dtype=np.int64))
        scores.append(result.scores)
        folds.append(np.full(len(test.y_codes), fold))
        fit_times.append(float(getattr(result, "fit_time", 0.0)))
        predict_times.append(float(getattr(result, "predict_time", 0.0)))
        reduction_times.append(float(getattr(result, "reduction_time", 0.0)))
    return {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "scores": np.vstack(scores),
        "fold": np.concatenate(folds),
        "fit_times": np.asarray(fit_times, dtype=np.float64),
        "predict_times": np.asarray(predict_times, dtype=np.float64),
        "reduction_times": np.asarray(reduction_times, dtype=np.float64),
    }


def load_or_run(cache: Path, dataset: str, key: str, winner: dict, retrain: bool) -> dict:
    path = cache / f"{dataset}_{key}.npz"
    if path.exists() and not retrain:
        loaded = dict(np.load(path))
        if "fit_times" in loaded and "predict_times" in loaded:
            return loaded
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


TECHNIQUES = ("none", "fcnn", "ennth", "drop", "del")


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


def format_time(seconds: float) -> str:
    """Format seconds into human-readable ns, µs, ms, or s."""
    if seconds <= 0:
        return "0 s"
    if seconds <= 1e-10:  # the bars are floored at 1e-10 s for the log scale (kNN fit)
        return "< 0.1 ns"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def grid_times(results: Path, dataset: str, winners: dict) -> dict[str, dict[str, np.ndarray]]:
    """Per-fold fit and predict times of each winner, read from the grid search CSVs.

    These are the runs behind the times of the report. The cached runs of this script also
    compute the ROC/PR scores inside ``predict``, which makes their predict times longer.
    """
    df = load_dataset(results, dataset)
    times = {}
    for key in WINNER_KEYS:
        if key not in winners:
            continue
        params = {k: str(v) for k, v in winners[key]["params"].items()}
        rows = df[
            (df["key"] == key.split("_")[0])
            & (df["reduction"] == params.get("reduction", "none"))
            & df["p"].map(lambda p, w=params: all(p.get(k) == v for k, v in w.items()))
        ].sort_values("fold")
        if rows["config"].nunique() != 1 or len(rows) != config.N_FOLDS:
            raise ValueError(f"{dataset} {key}: expected one configuration with 10 folds")
        times[key] = {
            "fit_times": rows["fit_time"].to_numpy(dtype=np.float64),
            "predict_times": rows["predict_time"].to_numpy(dtype=np.float64),
        }
    return times


def plot_timing_comparison(dataset: str, runs: dict, winners: dict, out: Path) -> None:
    """Compare per-instance training (fit) and inference (predict) times: eager vs lazy.

    ``runs`` maps each winner to its per-fold ``fit_times`` and ``predict_times`` (see
    ``grid_times``).
    """
    base_keys = [k for k in ("knn", "svm", "rf") if k in runs]
    red_keys = [k for k in ("knn_reduced", "svm_reduced", "rf_reduced") if k in runs]
    all_keys = [k for k in WINNER_KEYS if k in runs]

    # Pre-calculate training and test fold sizes to normalize per-instance
    train_sizes = []
    test_sizes = []
    for f in range(config.N_FOLDS):
        tr, te, _ = load_fold(dataset, f)
        train_sizes.append(len(tr.y))
        test_sizes.append(len(te.y_codes))
    train_sizes = np.asarray(train_sizes, dtype=np.float64)
    test_sizes = np.asarray(test_sizes, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax_base, ax_red), (ax_fit, ax_ratio) = axes

    c_fit = "#d95f02"  # Training / Fit
    c_pred = "#2a78d6"  # Inference / Predict

    # 1. Unreduced Models: Fit vs Predict (per-instance)
    x = np.arange(len(base_keys))
    width = 0.35
    fits = [max(float(np.mean(runs[k]["fit_times"] / train_sizes)), 1e-10) for k in base_keys]
    fit_stds = [float(np.std(runs[k]["fit_times"] / train_sizes)) for k in base_keys]
    preds = [max(float(np.mean(runs[k]["predict_times"] / test_sizes)), 1e-10) for k in base_keys]
    pred_stds = [float(np.std(runs[k]["predict_times"] / test_sizes)) for k in base_keys]

    b1 = ax_base.bar(
        x - width / 2,
        fits,
        width,
        yerr=fit_stds,
        label="Training (fit / train sample)",
        color=c_fit,
        edgecolor="white",
        capsize=3,
        error_kw={"ecolor": "#52514e", "elinewidth": 1},
    )
    b2 = ax_base.bar(
        x + width / 2,
        preds,
        width,
        yerr=pred_stds,
        label="Inference (predict / test query)",
        color=c_pred,
        edgecolor="white",
        capsize=3,
        error_kw={"ecolor": "#52514e", "elinewidth": 1},
    )

    for rect, val in zip(b1, fits, strict=False):
        ax_base.text(
            rect.get_x() + rect.get_width() / 2,
            val * 1.3,
            format_time(val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )
    for rect, val in zip(b2, preds, strict=False):
        ax_base.text(
            rect.get_x() + rect.get_width() / 2,
            val * 1.3,
            format_time(val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )

    ax_base.set_xticks(x)
    ax_base.set_xticklabels([NAMES[k] for k in base_keys])
    ax_base.set(
        yscale="log",
        ylabel="Time per instance (log scale)",
        title="Unreduced Winners: Training vs Inference Time (per instance)",
    )
    ax_base.legend(frameon=False, fontsize=9, loc="upper left")
    ax_base.grid(axis="x", visible=False)

    # 2. Reduced Models: Fit vs Predict (per-instance)
    if red_keys:
        x_red = np.arange(len(red_keys))
        fits_r = [max(float(np.mean(runs[k]["fit_times"] / train_sizes)), 1e-10) for k in red_keys]
        fit_stds_r = [float(np.std(runs[k]["fit_times"] / train_sizes)) for k in red_keys]
        preds_r = [
            max(float(np.mean(runs[k]["predict_times"] / test_sizes)), 1e-10) for k in red_keys
        ]
        pred_stds_r = [float(np.std(runs[k]["predict_times"] / test_sizes)) for k in red_keys]

        b1_r = ax_red.bar(
            x_red - width / 2,
            fits_r,
            width,
            yerr=fit_stds_r,
            label="Training (fit / train sample)",
            color=c_fit,
            edgecolor="white",
            capsize=3,
            error_kw={"ecolor": "#52514e", "elinewidth": 1},
        )
        b2_r = ax_red.bar(
            x_red + width / 2,
            preds_r,
            width,
            yerr=pred_stds_r,
            label="Inference (predict / test query)",
            color=c_pred,
            edgecolor="white",
            capsize=3,
            error_kw={"ecolor": "#52514e", "elinewidth": 1},
        )

        for rect, val in zip(b1_r, fits_r, strict=False):
            ax_red.text(
                rect.get_x() + rect.get_width() / 2,
                val * 1.3,
                format_time(val),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222",
            )
        for rect, val in zip(b2_r, preds_r, strict=False):
            ax_red.text(
                rect.get_x() + rect.get_width() / 2,
                val * 1.3,
                format_time(val),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222",
            )

        ax_red.set_xticks(x_red)
        ax_red.set_xticklabels([label(k, winners) for k in red_keys], rotation=15, ha="right")
        ax_red.set(
            yscale="log",
            ylabel="Time per instance (log scale)",
            title="Reduced Winners: Training vs Inference Time (per instance)",
        )
        ax_red.legend(frameon=False, fontsize=9, loc="upper left")
        ax_red.grid(axis="x", visible=False)

    # 3. Direct Training Time (Fit) Comparison (per-instance)
    x_all = np.arange(len(all_keys))
    all_fits = [max(float(np.mean(runs[k]["fit_times"] / train_sizes)), 1e-10) for k in all_keys]
    all_fit_stds = [float(np.std(runs[k]["fit_times"] / train_sizes)) for k in all_keys]
    bar_colors = [COLORS[k.replace("_reduced", "")] for k in all_keys]
    hatches = ["//" if k.endswith("_reduced") else "" for k in all_keys]

    bars_fit = ax_fit.bar(
        x_all,
        all_fits,
        yerr=all_fit_stds,
        color=bar_colors,
        edgecolor="#52514e",
        linewidth=0.8,
        capsize=3,
        error_kw={"ecolor": "#52514e", "elinewidth": 1},
    )
    for b, h in zip(bars_fit, hatches, strict=False):
        b.set_hatch(h)

    for rect, val in zip(bars_fit, all_fits, strict=False):
        ax_fit.text(
            rect.get_x() + rect.get_width() / 2,
            val * 1.35,
            format_time(val),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222",
        )

    ax_fit.set_xticks(x_all)
    ax_fit.set_xticklabels(
        [label(k, winners) for k in all_keys], rotation=25, ha="right", fontsize=8.5
    )
    ax_fit.set(
        yscale="log",
        ylabel="Fit time per training instance (s, log scale)",
        title="Training Time per Instance: Lazy (kNN) vs Eager (SVM, RF)",
    )
    ax_fit.grid(axis="x", visible=False)

    # 4. Training-to-Inference Ratio: (Fit / train sample) / (Predict / test sample)
    ratios = [
        float(np.mean(runs[k]["fit_times"] / train_sizes))
        / max(float(np.mean(runs[k]["predict_times"] / test_sizes)), 1e-12)
        for k in all_keys
    ]
    clamped_ratios = [max(r, 1e-6) for r in ratios]
    bars_ratio = ax_ratio.bar(
        x_all, clamped_ratios, color=bar_colors, edgecolor="#52514e", linewidth=0.8
    )
    for b, h in zip(bars_ratio, hatches, strict=False):
        b.set_hatch(h)

    for rect, val in zip(bars_ratio, ratios, strict=False):
        text_val = f"{val:.1f}x" if val >= 1.0 else f"1/{1.0 / max(val, 1e-12):.0f}x"
        ax_ratio.text(
            rect.get_x() + rect.get_width() / 2,
            max(val, 1e-6) * 1.3,
            text_val,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222",
        )

    ax_ratio.axhline(
        1.0,
        color="#8a8984",
        linestyle="--",
        linewidth=1.2,
        label="Equal per-sample time (Ratio = 1)",
    )
    ax_ratio.set_xticks(x_all)
    ax_ratio.set_xticklabels(
        [label(k, winners) for k in all_keys], rotation=25, ha="right", fontsize=8.5
    )
    ax_ratio.set(
        yscale="log",
        ylabel="Ratio: (Fit / train sample) / (Predict / test sample)",
        title="Per-Sample Cost Ratio: Eager (Fit >> Predict) vs Lazy (Predict >> Fit)",
    )
    ax_ratio.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_ratio.grid(axis="x", visible=False)

    # Adjust headroom
    for ax in (ax_base, ax_red, ax_fit, ax_ratio):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 0.3, ymax * 5.0)

    fig.suptitle(
        f"{dataset}: Per-Instance Training Time vs. Inference Time (Eager vs. Lazy Learners)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"timing_{dataset}.png", dpi=150)
    plt.close(fig)


def per_instance(seconds: float) -> str:
    """Time per instance in ns below 0.1 µs (the kNN fit), otherwise in µs."""
    return f"{seconds * 1e9:.2f} ns" if seconds < 1e-7 else f"{seconds * 1e6:.1f} µs"


def write_timing_and_winner_checks(results: Path, datasets: list[str]) -> Path:
    winners = json.loads((results / "winners.json").read_text())
    lines = [
        "# Timing and Winner Checks",
        "",
        "Generated by `experiments/plot_winners.py` from the official grid search results.",
        "",
        "### Time per instance of the best models (Table 7)",
        "",
        "fit_time / training fold size and predict_time / test fold size, mean over the 10 folds "
        "(base runs of the grid search).",
        "",
    ]
    time_rows = []
    rank_rows = []
    for dataset in datasets:
        df = load_dataset(results, dataset)
        train_sizes = np.array(
            [len(load_fold(dataset, f)[0].y) for f in range(config.N_FOLDS)], dtype=float
        )
        test_sizes = np.array(
            [len(load_fold(dataset, f)[1].y) for f in range(config.N_FOLDS)], dtype=float
        )
        times = grid_times(results, dataset, winners[dataset])
        accuracy = {}
        for model in ("knn", "svm", "rf"):
            fit = float(np.mean(times[model]["fit_times"] / train_sizes))
            pred = float(np.mean(times[model]["predict_times"] / test_sizes))
            time_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fit per train instance": per_instance(fit),
                    "predict per test instance": per_instance(pred),
                }
            )
            params = {k: str(v) for k, v in winners[dataset][model]["params"].items()}
            m_rows = df[
                (df["key"] == model)
                & (df["reduction"] == "none")
                & df["p"].map(lambda p, w=params: all(p.get(k) == v for k, v in w.items()))
            ].sort_values("fold")
            accuracy[model] = m_rows["accuracy"].to_numpy()
        ranks = stats.mean_ranks(pd.DataFrame(accuracy))
        rank_rows.append(
            {
                "dataset": dataset,
                "kNN": f"{ranks['knn']:.2f}",
                "SVM": f"{ranks['svm']:.2f}",
                "RF": f"{ranks['rf']:.2f}",
            }
        )

    lines.append(md_table(pd.DataFrame(time_rows)))
    lines.extend(["", "### Mean ranks of the best kNN, SVM and RF (accuracy, 10 folds)", ""])
    lines.append(md_table(pd.DataFrame(rank_rows)))

    if "splice" in datasets:
        df_splice = load_dataset(results, "splice")
        df_base_knn = df_splice[
            (df_splice["key"] == "knn") & (df_splice["reduction"] == "none")
        ].copy()
        df_base_knn["distance"] = df_base_knn["p"].map(lambda p: p["distance"])
        df_base_knn["others"] = df_base_knn["p"].map(
            lambda p: f"k={p['k']}, {p['voting']}, {p['retention']}"
        )
        table = df_base_knn.pivot_table(
            index=["others", "fold"], columns="distance", values="n_correct"
        )
        differ = table[table.max(axis=1) > table.min(axis=1)]
        means = df_base_knn.groupby("distance")["accuracy"].mean()
        n_others = df_base_knn["others"].nunique()
        lines.extend(
            [
                "",
                "### Splice: do the five distances give the same predictions?",
                "",
                f"- Cells (configuration x fold) where the distances disagree: {len(differ)} of "
                f"{len(table)}; combinations of the other parameters involved: "
                f"{differ.index.get_level_values(0).nunique()} of {n_others}.",
                "- Mean accuracy per distance (6 decimals): "
                + ", ".join(f"{d} {v:.6f}" for d, v in means.items())
                + ".",
                "",
            ]
        )
        if len(differ):
            lines.extend(["Correct test cases in those cells:", ""])
            shown = differ.astype(int).reset_index()
            lines.append(md_table(shown))

    out_path = results / "timing_report.md"
    out_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"Timing report saved to {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    default_results = config.RESULTS_DIR / "gridsearch"
    parser = argparse.ArgumentParser(description="Plots of the grid search winners.")
    parser.add_argument("--results", type=Path, default=default_results)
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
        plot_timing_comparison(dataset, grid_times(args.results, dataset, winners), winners, out)
        print(f"[{dataset}] plots saved to {out}")
    write_timing_and_winner_checks(args.results, datasets)


if __name__ == "__main__":
    main()
