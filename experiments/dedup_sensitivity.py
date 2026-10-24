"""Sensitivity of the splice results to the exact duplicate rows of the training folds.

splice has exact duplicates (same sequence, same class) inside every training fold, and some
test rows have an identical copy in their training fold. The official results keep them, because
the folds may not be modified. This script reruns the grid v2 winners of splice on the 10 folds
twice: as in the grid ("full") and with ``Preprocessor(drop_duplicate_rows=True)``, which removes
the exact copies from the TRAIN fold only ("dedup"). It does not change the official results.

Models: the best kNN, SVM and RF, and the best kNN with each reduction technique (best variant).
Per model and variant: accuracy, macro F1, storage, time. Full vs dedup: Wilcoxon (exact) with
Holm over all comparisons; the grid's own tests (kNN vs SVM vs RF, reduction vs none) are repeated
on both variants. Also: accuracy of the best kNN on the test rows that have an identical copy in
their training fold ("free answers") vs the rest.

Run from the project root:  python experiments/dedup_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import stats  # noqa: E402
from arff_parser import load_fold  # noqa: E402
from experiment_runner import confusion_matrix, macro_f1  # noqa: E402
from experiments.grid_search_v2 import Tester, parse_params  # noqa: E402
from models.knn.knn import KNN  # noqa: E402
from models.knn.reduction import reduce_with_cache  # noqa: E402
from models.random_forest import RandomForest  # noqa: E402
from models.svm import SVM  # noqa: E402
from preprocessing import Preprocessor  # noqa: E402

DATASET = "splice"
VARIANTS = {"full": False, "dedup": True}
MODEL_CLASSES = {"knn": KNN, "svm": SVM, "rf": RandomForest}
ALPHA = 0.05


def model_specs(winners: dict) -> dict[str, tuple[str, dict]]:
    """Label -> (model key, parameters) for the 7 models, from winners.json of splice."""
    specs = {
        "kNN": ("knn", dict(winners["knn"]["params"])),
        "SVM": ("svm", dict(winners["svm"]["params"])),
        "RF": ("rf", dict(winners["rf"]["params"])),
    }
    for technique, variant in winners["reduction_champions"].items():
        params = {**winners["knn"]["params"], "reduction": technique, **variant}
        specs[f"kNN+{technique}"] = ("knn", params)
    return specs


def official_accuracies(results: Path, key: str, params: dict) -> np.ndarray | None:
    """Per-fold accuracies of this configuration in the grid v2 CSVs (None if not found)."""
    stage = "base" if params.get("reduction", "none") == "none" else "reduction"
    wanted = {k: str(v) for k, v in params.items()}
    for path in sorted((results / DATASET / stage / key).glob("*.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        if parse_params(frame["params"].iloc[0]) == wanted:
            return frame.sort_values("fold")["accuracy"].to_numpy()
    return None


def load_variant(fold: int, dedup: bool):
    """Raw and preprocessed train/test of one fold, with or without the train duplicates."""
    train_raw, test_raw = load_fold(DATASET, fold)
    pre = Preprocessor.for_dataset(DATASET, drop_duplicate_rows=dedup)
    train = pre.fit_transform(train_raw)
    return train_raw, test_raw, train, pre.transform(test_raw), pre.n_duplicates_removed_


def run_model(key: str, params: dict, train, test, cache: Path | None) -> dict:
    """Fit and predict one fold; storage = % of the original training fold kept."""
    model = MODEL_CLASSES[key](dict(params))
    model.reduction_cache_dir = cache
    start = perf_counter()
    result = model.fit_predict(train, test)
    elapsed = perf_counter() - start
    # A cached reduction is charged with its measured time, as in experiment_runner.run_fold
    elapsed += float(result.reduction_time) - float(result.reduction_wall_time)
    predictions = np.asarray(result.predictions, dtype=np.int64)
    kept = len(train.y_codes)
    reducer = getattr(model, "reduction_instance", None)
    if reducer is not None and reducer.method != "none":
        kept = len(reduce_with_cache(reducer, train, cache)[0].y_codes)  # cache hit
    matrix = confusion_matrix(test.y_codes, predictions.tolist(), len(test.class_values))
    return {
        "predictions": predictions,
        "accuracy": float(np.mean(predictions == test.y_codes)),
        "macro_f1": macro_f1(matrix),
        "time": elapsed,
        "reduction_time": float(result.reduction_time),
        "kept": kept,
        "case_base": int(result.storage),
    }


def twin_mask(train_raw, test_raw) -> np.ndarray:
    """Test rows whose attributes are identical to some row of the raw training fold."""

    def key(data, i):
        return tuple(None if data.missing[i, j] else data.X[i, j] for j in range(data.n_features))

    rows = {key(train_raw, i) for i in range(train_raw.n_instances)}
    return np.array([key(test_raw, i) in rows for i in range(test_raw.n_instances)], dtype=bool)


def run_all(
    results: Path, cache: Path | None, folds=range(10)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold rows for every model and variant, and the free-answer rows of the best kNN."""
    winners = json.loads((results / "winners.json").read_text())[DATASET]
    specs = model_specs(winners)
    rows, free = [], []
    for variant, dedup in VARIANTS.items():
        for label, (key, params) in specs.items():
            config.set_seed()  # as run_experiment: one seed reset per 10-fold experiment
            for fold in folds:
                train_raw, test_raw, train, test, removed = load_variant(fold, dedup)
                out = run_model(key, params, train, test, cache)
                n_train = train_raw.n_instances
                rows.append(
                    {
                        "variant": variant,
                        "model": label,
                        "fold": fold,
                        "train_rows": len(train.y_codes),
                        "removed": removed,
                        "accuracy": out["accuracy"],
                        "macro_f1": out["macro_f1"],
                        "time": out["time"],
                        "reduction_time": out["reduction_time"],
                        "storage": 100.0 * out["kept"] / n_train,
                        "final_case_base": 100.0 * out["case_base"] / n_train,
                    }
                )
                print(f"  {variant:5} {label:10} fold {fold}: {out['accuracy']:.4f}", flush=True)
                if label == "kNN":
                    twins = twin_mask(train_raw, test_raw)
                    correct = out["predictions"] == test.y_codes
                    free.append(
                        {
                            "variant": variant,
                            "fold": fold,
                            "test_rows": len(correct),
                            "twin_rows": int(twins.sum()),
                            "twin_correct": int(correct[twins].sum()),
                            "rest_rows": int((~twins).sum()),
                            "rest_correct": int(correct[~twins].sum()),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(free)


def check_official(rows: pd.DataFrame, results: Path) -> list[str]:
    """Models whose 'full' accuracies differ from the grid v2 CSVs (empty list = all match)."""
    winners = json.loads((results / "winners.json").read_text())[DATASET]
    mismatches = []
    for label, (key, params) in model_specs(winners).items():
        ours = rows[(rows["variant"] == "full") & (rows["model"] == label)]
        official = official_accuracies(results, key, params)
        if official is None or not np.allclose(ours.sort_values("fold")["accuracy"], official):
            mismatches.append(label)
    return mismatches


def wide(rows: pd.DataFrame, variant: str, metric: str, models: list[str]) -> pd.DataFrame:
    part = rows[rows["variant"] == variant]
    return part.pivot(index="fold", columns="model", values=metric)[models]


def full_vs_dedup(rows: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon full vs dedup per model and metric, Holm over all of them."""
    records = []
    for metric in ("accuracy", "macro_f1"):
        for model in rows["model"].unique():
            table = pd.DataFrame({v: wide(rows, v, metric, [model])[model] for v in VARIANTS})
            result = stats.compare_two(table, "full", "dedup")
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "full": table["full"].mean(),
                    "dedup": table["dedup"].mean(),
                    "difference": table["dedup"].mean() - table["full"].mean(),
                    "folds_changed": int((table["full"] != table["dedup"]).sum()),
                    "p_raw": result.wilcoxon_p_value,
                }
            )
    frame = pd.DataFrame(records)
    frame["p_holm"] = stats.holm_adjust(frame["p_raw"]).to_numpy()
    frame["significant"] = frame["p_holm"] < ALPHA
    return frame


def grid_tests(rows: pd.DataFrame, figures: Path) -> pd.DataFrame:
    """The grid's tests for kNN vs SVM vs RF and for the reduction, on both variants."""
    records = []
    reductions = [m for m in rows["model"].unique() if m.startswith("kNN+")]
    for variant in VARIANTS:
        tester = Tester(DATASET, figures / variant)
        acc = wide(rows, variant, "accuracy", ["kNN", "SVM", "RF"])
        tester.compare(acc, "Models", "kNN vs SVM vs RF")
        tester.holm_pairs(acc, "kNN", "Models", "kNN vs SVM vs RF")
        red = wide(rows, variant, "accuracy", ["kNN", *reductions]).rename(columns={"kNN": "none"})
        tester.compare(red, "Reduction", "kNN reduction", control="none")
        for record in tester.records:
            records.append({"variant": variant, **record})
    return pd.DataFrame(records)


def md_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4g}" if abs(value) < 1e-3 and value != 0 else f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(False)]
    return "\n".join(lines)


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["model", "variant"], sort=False)
    table = grouped.agg(
        accuracy=("accuracy", "mean"),
        std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"),
        storage=("storage", "mean"),
        final_case_base=("final_case_base", "mean"),
        time=("time", "mean"),
    ).reset_index()
    table["final_case_base"] = [
        v if m.startswith("kNN") else float("nan")
        for m, v in zip(table["model"], table["final_case_base"], strict=True)
    ]
    return table


def free_table(free: pd.DataFrame) -> pd.DataFrame:
    def summarize(part: pd.DataFrame, label) -> dict:
        twin, rest = part["twin_rows"].sum(), part["rest_rows"].sum()
        return {
            "variant": part["variant"].iloc[0],
            "fold": label,
            "twin rows": int(twin),
            "accuracy on twins": part["twin_correct"].sum() / twin if twin else float("nan"),
            "rest rows": int(rest),
            "accuracy on the rest": part["rest_correct"].sum() / rest,
            "overall": (part["twin_correct"].sum() + part["rest_correct"].sum())
            / part["test_rows"].sum(),
        }

    out = []
    for variant, part in free.groupby("variant", sort=False):
        out += [summarize(part[part["fold"] == f], f) for f in sorted(part["fold"])]
        out.append(summarize(part, "total"))
    return pd.DataFrame(out)


def report(rows, free, tests, comparisons, mismatches) -> str:
    removed = rows[(rows["variant"] == "dedup") & (rows["model"] == "kNN")]
    free_full = free[free["variant"] == "full"]
    twins = free_full["twin_rows"].sum()
    total = free_full["test_rows"].sum()
    rest_acc = free_full["rest_correct"].sum() / free_full["rest_rows"].sum()
    overall = (free_full["twin_correct"].sum() + free_full["rest_correct"].sum()) / total
    without = rest_acc  # accuracy if the twin rows were answered like the rest
    shown = tests[["variant", "question", "test", "p_value", "significant", "post_hoc"]]
    return (
        "\n\n".join(
            [
                "# splice: sensitivity to the duplicate rows of the training folds",
                "Not an official result: the folds may not be modified (assignment, Section 1.1). "
                "`full` = as in grid v2; `dedup` = `Preprocessor(drop_duplicate_rows=True)`, exact "
                "copies (same sequence and class) removed from the training fold only. Storage = % of "
                "the original training fold kept; final case base = kNN case base after the test phase.",
                "Check: the `full` run reproduces the per-fold accuracies of grid v2 for "
                + ("every model." if not mismatches else f"all models except {mismatches}."),
                "Rows removed per fold: "
                + ", ".join(f"{r.fold}: {r.removed}" for r in removed.itertuples())
                + f" (range {removed['removed'].min()}-{removed['removed'].max()}, "
                f"{100 * removed['removed'].min() / (removed['train_rows'] + removed['removed']).min():.1f}"
                f"-{100 * removed['removed'].max() / (removed['train_rows'] + removed['removed']).max():.1f}%"
                " of the training fold).",
                "## Mean over the 10 folds",
                md_table(summary_table(rows)),
                "## Full vs dedup (Wilcoxon exact, Holm over all rows)",
                md_table(comparisons),
                "## The grid's tests on each variant",
                md_table(shown),
                "## Free answers of the best kNN",
                f"Test rows with an identical copy in their training fold: {twins} of {total} "
                f"({100 * twins / total:.1f}%). With `full`, the best kNN is right on "
                f"{free_full['twin_correct'].sum()} of them; on the other rows its accuracy is "
                f"{rest_acc:.4f}. Overall accuracy {overall:.4f}; if the twin rows were answered "
                f"like the rest it would be {without:.4f} ({overall - without:+.4f}).",
                md_table(free_table(free)),
            ]
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR / "gridsearch_v2")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "dedup_sensitivity")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not store the reduced training sets in <out>/reduction_cache",
    )
    args = parser.parse_args(argv)

    cache = None if args.no_cache else args.out / "reduction_cache"
    rows, free = run_all(args.results, cache)
    mismatches = check_official(rows, args.results)
    comparisons = full_vs_dedup(rows)
    tests = grid_tests(rows, args.out / "figures")

    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "splice_folds.csv", index=False, float_format="%.6f")
    free.to_csv(args.out / "splice_free_answers.csv", index=False)
    tests.to_csv(args.out / "splice_tests.csv", index=False)
    text = report(rows, free, tests, comparisons, mismatches)
    (args.out / "splice.md").write_text(text, encoding="utf-8")
    print(text)
    if mismatches:
        print(f"WARNING: 'full' does not reproduce grid v2 for {mismatches}")


if __name__ == "__main__":
    main()
