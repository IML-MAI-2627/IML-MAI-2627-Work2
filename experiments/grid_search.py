"""Grid search: every experiment of Work 2, with and without instance reduction.

Each dataset is handled on its own (splice and vowel are never mixed) and its 10 folds are
the blocks of every statistical test. The pipeline and the reasons behind each test are in
docs/DECISIONS.md ("Grid search"). Run from the project root:

    python experiments/grid_search.py --dataset all --stage all

Stages of ``run`` (each one needs the winners of the previous one):
  1. base           full grids without reduction: kNN (120), SVM and RF.
  2. knn-reduction  every reduction variant on the top ``--reduce-top`` kNN configurations.
  3. model-reduction  best SVM and best RF trained on the reduced set of each technique.
``evaluate`` reads every CSV, runs the tests and writes report.md, tests.csv, summary.csv,
winners.json and the CD diagrams.

One CSV per configuration (10 folds). A configuration whose CSV exists is skipped, so an
interrupted run resumes where it stopped. CSVs are written atomically (temp file + rename).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import stats  # noqa: E402
from experiment_runner import format_params, run_experiment, save_results  # noqa: E402
from models.knn.knn import KNN  # noqa: E402
from models.random_forest import RandomForest  # noqa: E402
from models.svm import SVM  # noqa: E402

ALPHA = 0.05
# solve_time = time - reduction_time: classifying the test fold, without the reduction cost.
HIGHER_IS_BETTER = {**stats.HIGHER_IS_BETTER, "solve_time": False}
TOP_GROUP_SIZE = 5  # configurations in the "is the winner distinguishable?" test

CLASSIFIERS = {
    "knn": (KNN, "kNN"),
    "svm": (SVM, "Support Vector Machine"),
    "rf": (RandomForest, "Random Forest"),
}
MODEL_KEY = {name: key for key, (_, name) in CLASSIFIERS.items()}
SHORT = {"knn": "kNN", "svm": "SVM", "rf": "RF"}

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
BASE_GRIDS = {
    "knn": {
        "k": [3, 5, 7],
        "distance": ["euclidean", "manhattan", "canberra", "clark", "heom"],
        "voting": ["majority", "inverse_distance"],
        "retention": ["never", "always", "different_class", "oblivion_by_goodness"],
    },
    "svm": {
        "C": [0.1, 1.0, 3.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "class_weight": ["None", "balanced"],
        "coef0": [-0.5, 0.0, 0.5],
        "degree": [2, 3, 5],
    },
    "rf": {
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, "None"],
        "max_features": ["sqrt", "None"],
        "min_samples_leaf": [1, 2],
        # One core per fit, like the kNN, so the times are comparable (see DECISIONS.md).
        "n_jobs": [1],
    },
}

# Factors whose levels answer a question of the assignment (Section 1.3).
FACTORS = {
    "knn": ["k", "distance", "voting", "retention"],
    "svm": ["kernel", "class_weight"],
    "rf": ["criterion", "n_estimators", "max_features"],
}

# Variants of each technique; the best one on the best kNN represents the technique.
# DROP4 is left out: our implementation is O(m * n^2 log n) in Python, too slow on splice.
REDUCTION_VARIANTS = {
    "fcnn": [{"version": 1}, {"version": 2}],  # condensed
    "ennth": [{"threshold": 0.0}, {"threshold": 0.5}, {"threshold": 0.7}],  # edited
    "drop": [{"version": 3}],  # hybrid
    "del": [{}],  # hybrid
}
VARIANT_KEYS = ("version", "threshold")
KNN_BASE_KEYS = ("k", "distance", "voting", "retention")


def expand(grid: dict) -> list[dict]:
    """Cartesian product of a grid."""
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def prune_svm(params: dict) -> dict:
    """Drop the SVM parameters that the kernel ignores (as SVM.parse_params does)."""
    p = dict(params)
    if p["kernel"] == "linear":
        p.pop("gamma", None)
    if p["kernel"] != "poly":
        p.pop("degree", None)
        p.pop("coef0", None)
    return p


def unique(configs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for params in configs:
        key = format_params(params)
        if key not in seen:
            seen.add(key)
            out.append(params)
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def job_path(out: Path, dataset: str, stage: str, model: str, params: dict) -> Path:
    """CSV of one configuration; the name is a hash of the requested parameters."""
    digest = hashlib.sha1(f"{model}|{format_params(params)}".encode()).hexdigest()[:12]
    return out / dataset / stage / model / f"{model}_{digest}.csv"


def run_job(job: tuple[str, str, dict, str]) -> str:
    """Run one configuration on the 10 folds and save its CSV (worker entry point)."""
    dataset, model, params, path = job
    cls, name = CLASSIFIERS[model]
    classifier = cls(dict(params))
    # <out>/<dataset>/<stage>/<model>/<file>.csv -> <out>/reduction_cache (shared by all jobs)
    classifier.reduction_cache_dir = Path(path).parents[3] / "reduction_cache"
    results = run_experiment(dataset, classifier.fit_predict, name, classifier.params)
    tmp = Path(path).with_suffix(".tmp")
    save_results(results, tmp)
    tmp.replace(path)
    return path


def run_jobs(jobs: list[tuple[str, str, dict, str]], n_jobs: int, label: str) -> None:
    pending = [job for job in jobs if not Path(job[3]).exists()]
    print(f"[{label}] {len(jobs)} configurations, {len(pending)} to run")
    for job in pending:
        Path(job[3]).parent.mkdir(parents=True, exist_ok=True)
    if n_jobs == 1:
        for i, job in enumerate(pending, 1):
            run_job(job)
            print(f"  {i}/{len(pending)} {job[1]} {format_params(job[2])}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, _ in enumerate(pool.map(run_job, pending), 1):
            print(f"  {i}/{len(pending)}", flush=True)


def base_jobs(out: Path, dataset: str) -> list:
    jobs = []
    for model, grid in BASE_GRIDS.items():
        configs = expand(grid)
        if model == "svm":
            configs = unique([prune_svm(p) for p in configs])
        for params in configs:
            jobs.append(
                (dataset, model, params, str(job_path(out, dataset, "base", model, params)))
            )
    return jobs


def knn_reduction_jobs(out: Path, dataset: str, bases: list[dict]) -> list:
    jobs = []
    for base in bases:
        for technique, variants in REDUCTION_VARIANTS.items():
            for variant in variants:
                params = {**base, "reduction": technique, **variant}
                path = job_path(out, dataset, "reduction", "knn", params)
                jobs.append((dataset, "knn", params, str(path)))
    return jobs


def reduction_settings(best_knn: dict, technique: str, variant: dict) -> dict:
    """Reduction parameters for SVM/RF: the same reduced set as the reduced best kNN."""
    return {
        "reduction": technique,
        "reduction_k": best_knn["k"],
        "reduction_distance": best_knn["distance"],
        "reduction_voting": best_knn["voting"],
        **variant,
    }


def model_reduction_jobs(out, dataset, best: dict, champions: dict) -> list:
    jobs = []
    for model in ("svm", "rf"):
        for technique, variant in champions.items():
            params = {**best[model], **reduction_settings(best["knn"], technique, variant)}
            path = job_path(out, dataset, "reduction", model, params)
            jobs.append((dataset, model, params, str(path)))
    return jobs


# ---------------------------------------------------------------------------
# Reading results and selecting winners (selection rule fixed a priori:
# highest mean accuracy; ties -> lower mean time -> config name)
# ---------------------------------------------------------------------------
def parse_params(text: str) -> dict:
    if not isinstance(text, str) or not text:
        return {}
    return dict(item.split("=", 1) for item in text.split(";"))


def load_dataset(out: Path, dataset: str) -> pd.DataFrame:
    # Only the per-configuration CSVs (summary.csv and tests.csv live one level up).
    files = sorted(
        f for stage in ("base", "reduction") for f in (out / dataset / stage).rglob("*.csv")
    )
    if not files:
        raise FileNotFoundError(f"No results under {out / dataset}")
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    df["key"] = df["model"].map(MODEL_KEY)
    df = df[df["key"].notna()].copy()  # only kNN, SVM and RF take part in the analysis
    df["config"] = [stats.config_name(m, p) for m, p in zip(df["model"], df["params"], strict=True)]
    df["p"] = df["params"].map(parse_params)
    df["reduction"] = [p.get("reduction", "none") for p in df["p"]]
    if "macro_f1" not in df:
        df["macro_f1"] = float("nan")
    if "reduction_time" not in df:
        df["reduction_time"] = 0.0
    if "fit_time" not in df:
        df["fit_time"] = float("nan")
    if "predict_time" not in df:
        df["predict_time"] = float("nan")
    df["solve_time"] = df["time"] - df["reduction_time"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, best first."""
    summary = df.groupby("config", sort=False).agg(
        key=("key", "first"),
        params=("params", "first"),
        reduction=("reduction", "first"),
        folds=("fold", "nunique"),
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"),
        time=("time", "mean"),
        solve_time=("solve_time", "mean"),
        fit_time=("fit_time", "mean"),
        predict_time=("predict_time", "mean"),
        storage=("storage", "mean"),
    )
    summary = summary.reset_index()
    return summary.sort_values(
        ["accuracy", "time", "config"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)


def best_of(summary: pd.DataFrame, n: int = 1) -> list[str]:
    return summary["config"].head(n).tolist()


def params_of(df: pd.DataFrame, config_label: str) -> dict:
    return dict(df.loc[df["config"] == config_label, "p"].iloc[0])


def base_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df[(df["key"] == key) & (df["reduction"] == "none")]


def select_base_winners(df: pd.DataFrame, top: int) -> dict:
    """Best kNN (top-n list), best SVM and best RF on the full training set."""
    winners = {}
    for key in ("knn", "svm", "rf"):
        configs = best_of(summarize(base_rows(df, key)), top if key == "knn" else 1)
        winners[key] = [params_of(df, c) for c in configs]
    return winners


def matches(p: dict, base: dict, keys: tuple[str, ...]) -> bool:
    return all(str(p.get(k)) == str(base.get(k)) for k in keys)


def reduction_champions(df: pd.DataFrame, best_knn: dict) -> dict[str, dict]:
    """Best variant of each technique on the best kNN (mean accuracy, then time)."""
    champions = {}
    for technique in REDUCTION_VARIANTS:
        rows = df[
            (df["key"] == "knn")
            & (df["reduction"] == technique)
            & df["p"].map(lambda p: matches(p, best_knn, KNN_BASE_KEYS))
        ]
        if rows.empty:
            continue
        p = params_of(rows, best_of(summarize(rows))[0])
        champions[technique] = {k: p[k] for k in VARIANT_KEYS if k in p}
    return champions


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def score_table(df: pd.DataFrame, columns: dict[str, str], metric: str) -> pd.DataFrame:
    """Folds x configurations table with short column labels ({label: config})."""
    rows = df[df["config"].isin(columns.values())]
    table = rows.pivot_table(index="fold", columns="config", values=metric, aggfunc="first")
    table = table[list(columns.values())]
    table.columns = list(columns)
    return table.dropna(axis=0)


def label_of(df: pd.DataFrame, config_label: str) -> str:
    """Readable short name for CD diagrams."""
    row = df[df["config"] == config_label].iloc[0]
    p = row["p"]
    keys = {
        "knn": [*KNN_BASE_KEYS, "reduction", *VARIANT_KEYS],
        "svm": ["kernel", "C", "gamma", "degree", "coef0", "class_weight", "reduction"],
        "rf": ["criterion", "n_estimators", "max_depth", "max_features", "min_samples_leaf"],
    }.get(row["key"], [])
    parts = [str(p[k]) for k in keys if k in p and not (k == "reduction" and p[k] == "none")]
    return f"{SHORT.get(row['key'], row['model'])}({','.join(parts)})"


class Tester:
    """Runs the right test for k configurations and collects one record per test."""

    def __init__(self, dataset: str, figures: Path) -> None:
        self.dataset = dataset
        self.figures = figures
        self.records: list[dict] = []

    def compare(
        self,
        table: pd.DataFrame,
        section: str,
        question: str,
        metric: str = "accuracy",
        control: str | None = None,
    ) -> dict:
        """Wilcoxon for 2 configs; Friedman (Iman-Davenport) + Nemenyi / Bonferroni-Dunn."""
        higher = HIGHER_IS_BETTER[metric]
        means = table.mean().sort_values(ascending=not higher)
        record = {
            "dataset": self.dataset,
            "section": section,
            "question": question,
            "metric": metric,
            "k": table.shape[1],
            "n_folds": table.shape[0],
            "means": "; ".join(f"{c}={v:.4f}" for c, v in means.items()),
            "test": "",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "winner": means.index[0] if len(means) else "",
            "post_hoc": "",
            "figure": "",
        }
        if table.shape[1] < 2 or table.shape[0] < 2:
            record["test"] = "none (fewer than 2 configurations)"
        elif table.shape[1] == 2:
            first, second = table.columns
            result = stats.compare_two(table, first, second, higher_is_better=higher)
            record.update(
                test="Wilcoxon signed-ranks (exact)",
                statistic=result.wilcoxon_t,
                p_value=result.wilcoxon_p_value,
                significant=result.wilcoxon_p_value < ALPHA,
                post_hoc=f"R+={result.r_plus:g}, R-={result.r_minus:g}; "
                f"paired t-test p={result.t_p_value:.3g} (reference)",
            )
        else:
            friedman = stats.friedman_test(table, higher_is_better=higher)
            record.update(
                test="Friedman + Iman-Davenport",
                statistic=friedman.iman_davenport,
                p_value=friedman.iman_davenport_p_value,
                significant=friedman.iman_davenport_p_value < ALPHA,
                winner=friedman.mean_ranks.index[0],
            )
            if record["significant"]:
                record.update(self._post_hoc(table, question, metric, control, higher))
        self.records.append(record)
        return record

    def _post_hoc(self, table, question, metric, control, higher) -> dict:
        if control is not None:
            result = stats.bonferroni_dunn_test(table, control, ALPHA, higher_is_better=higher)
            different = result.rank_differences[result.significant]
            detail = ", ".join(
                f"{c} {'better' if d > 0 else 'worse'}" for c, d in different.items()
            )
            text = f"Bonferroni-Dunn vs {control} (CD={result.critical_difference:.2f}): "
            text += detail or "no configuration differs from the control"
        else:
            result = stats.nemenyi_test(table, ALPHA, higher_is_better=higher)
            best = result.mean_ranks.index[0]
            tied = [c for c in result.mean_ranks.index if not result.significant.loc[best, c]]
            text = f"Nemenyi (CD={result.critical_difference:.2f}): not different from best: "
            text += ", ".join(tied)
        name = "".join(ch if ch.isalnum() else "_" for ch in f"{question}_{metric}").strip("_")
        path = self.figures / f"cd_{self.dataset}_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = stats.plot_cd_diagram(
            result, title=f"{self.dataset}: {question} ({metric})", path=path
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {"post_hoc": text, "figure": str(path.name)}

    def holm_pairs(self, table: pd.DataFrame, control: str, section: str, question: str) -> None:
        """Control vs each other column with Wilcoxon, Holm-corrected (Demšar 2006, 3.2.2)."""
        raw = {}
        for other in table.columns.drop(control):
            result = stats.compare_two(table, control, other)
            raw[other] = result.wilcoxon_p_value
        adjusted = stats.holm_adjust(pd.Series(raw))
        for other, p_adj in adjusted.items():
            means = table[[control, other]].mean()
            self.records.append(
                {
                    "dataset": self.dataset,
                    "section": section,
                    "question": f"{question}: {control} vs {other}",
                    "metric": "accuracy",
                    "k": 2,
                    "n_folds": len(table),
                    "means": "; ".join(f"{c}={v:.4f}" for c, v in means.items()),
                    "test": "Wilcoxon signed-ranks + Holm",
                    "statistic": float("nan"),
                    "p_value": p_adj,
                    "significant": p_adj < ALPHA,
                    "winner": means.idxmax(),
                    "post_hoc": f"raw p={raw[other]:.4g}",
                    "figure": "",
                }
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def md_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def cell(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("|" + "---|" * len(frame.columns))
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(False)]
    return "\n".join(lines)


def level_order(level: str) -> tuple:
    """Sort 'n_estimators=50' before 'n_estimators=100'."""
    value = level.split("=", 1)[1]
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value)


def factor_analysis(df, tester: Tester, key: str, report: list[str]) -> None:
    """Per factor: main effect (no selection) and best configuration per level."""
    rows = base_rows(df, key)
    summary = summarize(rows)
    summary["p"] = summary["params"].map(parse_params)
    for factor in FACTORS[key]:
        rows_f = rows.assign(level=rows["p"].map(lambda p, f=factor: f"{f}={p.get(f)}"))
        levels = sorted(rows_f["level"].unique(), key=level_order)
        # (a) Main effect: per fold, mean over every configuration of the level.
        main = rows_f.groupby(["fold", "level"])["accuracy"].mean().unstack()[levels]
        tester.compare(main, SHORT[key], f"{SHORT[key]} {factor} main effect")
        # (b) Best configuration of each level (selected by mean accuracy).
        champions = {}
        for level in levels:
            in_level = summary[summary["p"].map(lambda p, f=factor: f"{f}={p.get(f)}") == level]
            champions[level] = in_level["config"].iloc[0]
        tester.compare(
            score_table(df, champions, "accuracy"), SHORT[key], f"{SHORT[key]} {factor} best"
        )
        level_rows = []
        for level in levels:
            in_level = rows_f[rows_f["level"] == level]
            best = summary[summary["config"] == champions[level]].iloc[0]
            level_rows.append(
                {
                    "level": level,
                    "configs": in_level["config"].nunique(),
                    "mean acc (all configs)": in_level["accuracy"].mean(),
                    "best acc": best["accuracy"],
                    "mean time (s)": in_level["time"].mean(),
                    "mean storage (%)": in_level["storage"].mean(),
                }
            )
        report += [f"#### {SHORT[key]}: {factor}", "", md_table(pd.DataFrame(level_rows)), ""]


def top_group(df, tester: Tester, key: str, summary: pd.DataFrame, report: list[str]) -> None:
    """Is the winner distinguishable from the next best configurations?"""
    configs = best_of(summary, TOP_GROUP_SIZE)
    columns = {f"#{i + 1} {label_of(df, c)}": c for i, c in enumerate(configs)}
    tester.compare(score_table(df, columns, "accuracy"), SHORT[key], f"{SHORT[key]} top-5")
    shown = summary.head(10)[["config", "accuracy", "accuracy_std", "macro_f1", "time", "storage"]]
    report += [f"#### {SHORT[key]}: 10 best configurations", "", md_table(shown), ""]


def evaluate(out: Path, dataset: str, top: int) -> dict:
    df = load_dataset(out, dataset)
    summary_all = summarize(df)
    summary_all.to_csv(out / dataset / "summary.csv", index=False)
    figures = out / dataset / "figures"
    tester = Tester(dataset, figures)
    report = [f"## {dataset}", ""]
    winners: dict = {}

    # 1-3. Each model on its own (no reduction).
    for key in ("knn", "svm", "rf"):
        report += [f"### {SHORT[key]} without reduction", ""]
        summary = summarize(base_rows(df, key))
        if summary.empty:
            continue
        factor_analysis(df, tester, key, report)
        top_group(df, tester, key, summary, report)
        winners[key] = summary["config"].iloc[0]

    # 4. Cross-model without reduction: best kNN vs best SVM vs best RF.
    names = {SHORT[k]: winners[k] for k in ("knn", "svm", "rf") if k in winners}
    if len(names) == 3:
        report += ["### Best kNN vs best SVM vs best RF (no reduction)", ""]
        acc = score_table(df, names, "accuracy")
        tester.compare(acc, "Models", "kNN vs SVM vs RF")
        tester.holm_pairs(acc, "kNN", "Models", "kNN vs SVM vs RF")
        tester.compare(score_table(df, names, "time"), "Models", "kNN vs SVM vs RF", "time")
        rows = summary_all[summary_all["config"].isin(names.values())]
        report += [md_table(rows[["config", "accuracy", "macro_f1", "time", "storage"]]), ""]

    # 5. Reduction on the best kNN.
    reduced = {}
    if "knn" in winners:
        best_knn = params_of(df, winners["knn"])
        champions = reduction_champions(df, best_knn)
        if champions:
            report += ["### Instance reduction on the best kNN", ""]
            reduced = reduction_section(df, tester, winners["knn"], champions, top, report)
            winners["reduction_champions"] = champions

    # 6. Reduction on the best SVM and the best RF, and models compared under each technique.
    if reduced:
        labels = {k: winners[k] for k in ("knn", "svm", "rf") if k in winners}
        model_reduction_section(df, tester, labels, winners["reduction_champions"], reduced, report)

    tests = pd.DataFrame(tester.records)
    tests.to_csv(out / dataset / "tests.csv", index=False)
    report += ["### All tests", "", tests_markdown(tests), ""]
    (out / dataset / "report.md").write_text("\n".join(report), encoding="utf-8")
    return winners_json(df, winners, reduced)


def reduction_section(df, tester, best_label, champions, top, report) -> dict[str, str]:
    """Reduction techniques on the best kNN, robustness on the top-n, retention vs reduction."""
    best_knn = params_of(df, best_label)
    knn_red = df[(df["key"] == "knn") & (df["reduction"] != "none")]
    on_best = knn_red[knn_red["p"].map(lambda p: matches(p, best_knn, KNN_BASE_KEYS))]
    shown = summarize(on_best)[["config", "accuracy", "macro_f1", "time", "solve_time", "storage"]]
    report += ["All variants on the best kNN:", "", md_table(shown), ""]

    columns = {"none": best_label}
    for technique, variant in champions.items():
        wanted = {**best_knn, "reduction": technique, **variant}
        columns[technique] = config_matching(on_best, wanted)
    for metric in ("accuracy", "solve_time"):
        tester.compare(
            score_table(df, columns, metric), "Reduction", "kNN reduction", metric, control="none"
        )
    table = summarize(df[df["config"].isin(columns.values())])
    table.insert(0, "technique", table["config"].map({v: k for k, v in columns.items()}))
    report += [
        md_table(table[["technique", "accuracy", "macro_f1", "time", "solve_time", "storage"]]),
        "",
    ]

    # Robustness: the same technique variants on the top-n kNN configurations (descriptive).
    tops = best_of(summarize(base_rows(df, "knn")), top)
    rows = []
    for rank, cfg in enumerate(tops, 1):
        base = params_of(df, cfg)
        row = {"rank": rank, "kNN": label_of(df, cfg), "none": cell_of(df, cfg)}
        for technique, variant in champions.items():
            match = config_matching(knn_red, {**base, "reduction": technique, **variant})
            if match:
                row[technique] = cell_of(df, match)
        rows.append(row)
    report += ["Top kNN configurations with each technique, accuracy (storage):", ""]
    report += [md_table(pd.DataFrame(rows)), ""]

    # Retention vs reduction: best config of each retention policy vs reduced best kNN.
    summary = summarize(base_rows(df, "knn"))
    retention = {}
    for policy in BASE_GRIDS["knn"]["retention"]:
        in_policy = summary[
            summary["params"].map(lambda t, p=policy: parse_params(t).get("retention") == p)
        ]
        if not in_policy.empty:
            retention[f"ret={policy}"] = in_policy["config"].iloc[0]
    mixed = {**retention, **{f"red={t}": c for t, c in columns.items() if t != "none"}}
    tester.compare(score_table(df, mixed, "accuracy"), "Reduction", "retention vs reduction")
    table = summarize(df[df["config"].isin(mixed.values())])
    table.insert(0, "variant", table["config"].map({v: k for k, v in mixed.items()}))
    report += ["Retention vs reduction:", ""]
    report += [md_table(table[["variant", "accuracy", "time", "solve_time", "storage"]]), ""]
    return columns


def model_reduction_section(df, tester, labels, champions, knn_columns, report) -> None:
    """Reduction on the best SVM / RF, then the three models compared under each technique."""
    best = {k: params_of(df, c) for k, c in labels.items()}
    per_technique = {t: {"kNN": c} for t, c in knn_columns.items()}
    for key in ("svm", "rf"):
        if key not in best:
            continue
        rows = df[df["key"] == key]
        columns = {"none": labels[key]}
        for technique, variant in champions.items():
            wanted = {**best[key], **reduction_settings(best["knn"], technique, variant)}
            match = config_matching(rows, wanted)
            if match:
                columns[technique] = match
        for technique, cfg in columns.items():
            per_technique.setdefault(technique, {})[SHORT[key]] = cfg
        report += [f"### Instance reduction on the best {SHORT[key]}", ""]
        for metric in ("accuracy", "solve_time"):
            tester.compare(
                score_table(df, columns, metric),
                "Reduction",
                f"{SHORT[key]} reduction",
                metric,
                control="none",
            )
        table = summarize(df[df["config"].isin(columns.values())])
        table.insert(0, "technique", table["config"].map({v: k for k, v in columns.items()}))
        report += [
            md_table(table[["technique", "accuracy", "macro_f1", "time", "solve_time", "storage"]]),
            "",
        ]

    report += ["### kNN vs SVM vs RF with each reduction technique", ""]
    for technique, columns in per_technique.items():
        if technique == "none" or len(columns) < 3:
            continue
        tester.compare(score_table(df, columns, "accuracy"), "Models", f"models with {technique}")
        tester.compare(
            score_table(df, columns, "time"), "Models", f"models with {technique}", "time"
        )


def cell_of(df: pd.DataFrame, cfg: str) -> str:
    """'accuracy (storage%)' of one configuration, for the robustness table."""
    s = summarize(df[df["config"] == cfg]).iloc[0]
    return f"{s.accuracy:.4f} ({s.storage:.0f}%)"


def config_matching(rows: pd.DataFrame, wanted: dict) -> str | None:
    """Config label of the row whose parameters contain all of ``wanted``."""
    wanted = {k: str(v) for k, v in wanted.items()}
    hit = rows[rows["p"].map(lambda p: all(p.get(k) == v for k, v in wanted.items()))]
    if hit.empty:
        return None
    return summarize(hit)["config"].iloc[0]


def tests_markdown(tests: pd.DataFrame) -> str:
    if tests.empty:
        return "(no tests)"
    shown = tests[["section", "question", "metric", "k", "test", "p_value", "significant"]].copy()
    shown["winner / post-hoc"] = [
        f"{w}. {p}" if p else str(w)
        for w, p in zip(tests["winner"], tests["post_hoc"], strict=True)
    ]
    shown["p_value"] = shown["p_value"].map(lambda v: f"{v:.3g}")
    return md_table(shown)


def winners_json(df: pd.DataFrame, winners: dict, reduced_knn: dict) -> dict:
    """Parameters of the winners, for experiments/plot_winners.py."""
    out = {}
    for key in ("knn", "svm", "rf"):
        if key in winners:
            cfg = winners[key]
            out[key] = {
                "model": key,
                "params": params_of(df, cfg),
                "accuracy": summarize(df[df["config"] == cfg])["accuracy"].iloc[0],
            }
    if reduced_knn and "knn" in out:
        reductions = {t: c for t, c in reduced_knn.items() if t != "none"}
        best_t, best_cfg = max(
            reductions.items(),
            key=lambda item: summarize(df[df["config"] == item[1]])["accuracy"].iloc[0],
        )
        out["best_reduction"] = best_t
        out["reduction_champions"] = winners.get("reduction_champions", {})
        out["knn_reduced"] = {
            "model": "knn",
            "params": params_of(df, best_cfg),
            "accuracy": summarize(df[df["config"] == best_cfg])["accuracy"].iloc[0],
        }
        for key in ("svm", "rf"):
            if key not in out:
                continue
            wanted = {
                **out[key]["params"],
                **reduction_settings(
                    out["knn"]["params"], best_t, winners["reduction_champions"][best_t]
                ),
            }
            cfg = config_matching(df[df["key"] == key], wanted)
            if cfg:
                out[f"{key}_reduced"] = {
                    "model": key,
                    "params": params_of(df, cfg),
                    "accuracy": summarize(df[df["config"] == cfg])["accuracy"].iloc[0],
                }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(out: Path, dataset: str, n_jobs: int, top: int) -> None:
    run_jobs(base_jobs(out, dataset), n_jobs, f"{dataset} base")
    df = load_dataset(out, dataset)
    winners = select_base_winners(df, top)
    run_jobs(knn_reduction_jobs(out, dataset, winners["knn"]), n_jobs, f"{dataset} kNN reduction")
    df = load_dataset(out, dataset)
    champions = reduction_champions(df, winners["knn"][0])
    best = {"knn": winners["knn"][0], "svm": winners["svm"][0], "rf": winners["rf"][0]}
    jobs = model_reduction_jobs(out, dataset, best, champions)
    run_jobs(jobs, n_jobs, f"{dataset} SVM/RF reduction")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grid search with and without reduction.")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument("--stage", choices=["run", "evaluate", "all"], default="all")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "gridsearch")
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel processes (1 keeps the times clean)"
    )
    parser.add_argument(
        "--reduce-top", type=int, default=3, help="kNN configurations to reduce (1 = best only)"
    )
    args = parser.parse_args(argv)

    datasets = list(config.DATASETS) if args.dataset == "all" else [args.dataset]
    all_winners = {}
    winners_path = args.out / "winners.json"
    if winners_path.exists():
        all_winners = json.loads(winners_path.read_text())
    for dataset in datasets:
        if args.stage in ("run", "all"):
            run(args.out, dataset, args.jobs, args.reduce_top)
        if args.stage in ("evaluate", "all"):
            all_winners[dataset] = evaluate(args.out, dataset, args.reduce_top)
            print(f"[{dataset}] report: {args.out / dataset / 'report.md'}")
    if args.stage in ("evaluate", "all"):
        winners_path.parent.mkdir(parents=True, exist_ok=True)
        winners_path.write_text(json.dumps(all_winners, indent=2, default=str))
        print(f"Winners saved to {winners_path}")


if __name__ == "__main__":
    main()
