"""Which vowel columns to keep: the best vowel kNN with three column sets.

(A) only the 10 numeric features, (B) A + Sex (the current preprocessing, DEFAULT_DROP),
(C) B + Speaker_Number. Only the dropped columns change; the kNN is the best vowel kNN of
grid v2 (read from winners.json). Writes a Markdown report and the per-fold accuracies.

Run from the project root:  python experiments/vowel_columns.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import stats  # noqa: E402
from arff_parser import load_fold  # noqa: E402
from models.knn.distances import heom  # noqa: E402
from models.knn.knn import KNN  # noqa: E402
from preprocessing import DEFAULT_DROP, Preprocessor  # noqa: E402

DATASET = "vowel"
VARIANTS: dict[str, tuple[str, ...]] = {
    "A: 10 numeric": ("Train_or_Test", "Speaker_Number", "Sex"),
    "B: + Sex (current)": DEFAULT_DROP[DATASET],
    "C: + Sex + Speaker_Number": ("Train_or_Test",),
}
ALPHA = 0.05


def best_vowel_knn(results: Path) -> dict:
    """Parameters of the best vowel kNN of grid v2."""
    path = results / "winners.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run grid_search_v2.py --stage evaluate first")
    return dict(json.loads(path.read_text())[DATASET]["knn"]["params"])


def run_variants(params: dict, folds=range(10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold accuracy and storage (%) of the kNN for each column set (columns = variants)."""
    accuracy, storage = {}, {}
    for name, drop in VARIANTS.items():
        config.set_seed()
        model = KNN(dict(params))
        accs, stor = [], []
        for fold in folds:
            train_raw, test_raw = load_fold(DATASET, fold)
            pre = Preprocessor(drop=drop).fit(train_raw)
            train, test = pre.transform(train_raw), pre.transform(test_raw)
            result = model.fit_predict(train, test)
            accs.append(float(np.mean(np.asarray(result.predictions) == test.y_codes)))
            stor.append(100.0 * result.storage / len(train.y_codes))
        accuracy[name], storage[name] = accs, stor
    index = pd.Index(list(folds), name="fold")
    return pd.DataFrame(accuracy, index=index), pd.DataFrame(storage, index=index)


def same_speaker_neighbours(folds=range(10)) -> tuple[int, int, int]:
    """With the 10 numeric features only: test cases whose nearest training case (HEOM) is
    the same speaker, the same speaker and vowel, and the total."""
    same_speaker = same_both = total = 0
    for fold in folds:
        train_raw, test_raw = load_fold(DATASET, fold)
        pre = Preprocessor(drop=VARIANTS["A: 10 numeric"]).fit(train_raw)
        train, test = pre.transform(train_raw), pre.transform(test_raw)
        speaker = list(train_raw.attribute_names).index("Speaker_Number")
        nearest = np.argmin(heom(train, test), axis=1)
        same = train_raw.X[nearest, speaker] == test_raw.X[:, speaker]
        same_speaker += int(np.sum(same))
        same_both += int(np.sum(same & (train.y_codes[nearest] == test.y_codes)))
        total += len(test.y_codes)
    return same_speaker, same_both, total


def md_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(False)]
    return "\n".join(lines)


def report(params: dict, accuracy: pd.DataFrame, storage: pd.DataFrame) -> str:
    """Markdown with the summary, the Friedman test and the pairwise Wilcoxon tests (Holm)."""
    names = list(VARIANTS)
    ranks = stats.mean_ranks(accuracy)
    summary = pd.DataFrame(
        {
            "Variant": names,
            "Mean accuracy": [f"{accuracy[c].mean():.4f}" for c in names],
            "Std": [f"{accuracy[c].std():.4f}" for c in names],
            "Mean rank": [f"{ranks[c]:.2f}" for c in names],
            "Storage (%)": [f"{storage[c].mean():.1f}" for c in names],
        }
    )
    friedman = stats.friedman_test(accuracy)
    p = friedman.iman_davenport_p_value
    verdict = "significant" if p < ALPHA else "not significant"
    pairs = [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]
    raw = pd.Series(
        {f"{a} vs {b}": stats.compare_two(accuracy, a, b).wilcoxon_p_value for a, b in pairs}
    )
    adjusted = stats.holm_adjust(raw)
    tests = pd.DataFrame(
        {
            "Pair": raw.index,
            "Mean difference (second - first)": [
                f"{accuracy[b].mean() - accuracy[a].mean():+.4f}" for a, b in pairs
            ],
            "Wilcoxon p": [f"{v:.4f}" for v in raw],
            "Holm-adjusted p": [f"{v:.4f}" for v in adjusted],
            "Significant": ["yes" if v < ALPHA else "no" for v in adjusted],
        }
    )
    folds = accuracy.round(4).reset_index()
    return "\n\n".join(
        [
            "# vowel: which columns to keep",
            f"Best vowel kNN of grid v2: `{params}`. Only the dropped columns change.",
            md_table(summary),
            f"Friedman + Iman-Davenport over the {len(accuracy)} folds: "
            f"F_F = {friedman.iman_davenport:.3f}, p = {p:.4g} ({verdict} at alpha = {ALPHA}).",
            md_table(tests),
            "Per-fold accuracy:",
            md_table(folds),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR / "gridsearch_v2")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "vowel_columns")
    args = parser.parse_args(argv)

    params = best_vowel_knn(args.results)
    accuracy, storage = run_variants(params)
    text = report(params, accuracy, storage)
    same_speaker, same_both, total = same_speaker_neighbours()
    text += (
        f"\n\nWith the 10 numeric features only (variant A, HEOM), the nearest training case of a "
        f"test case is the same speaker in {same_speaker} of {total} test cases "
        f"({100 * same_speaker / total:.1f}%), and the same speaker and vowel in {same_both} "
        f"({100 * same_both / total:.1f}%).\n"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vowel_columns.md").write_text(text, encoding="utf-8")
    accuracy.to_csv(args.out / "vowel_columns_accuracy.csv")
    print(text)
    print(f"saved {args.out / 'vowel_columns.md'}")


if __name__ == "__main__":
    main()
