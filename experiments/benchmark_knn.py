"""Compare knn_base (pairwise distance) with knn_fast (neighbours computed once) on every fold.

Run from the project root:
    python experiments/benchmark_knn.py --dataset splice
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import knn_base  # noqa: E402
import knn_fast  # noqa: E402
from preprocessing import preprocess_fold  # noqa: E402

K_VALUES = (3, 5, 7)


def majority_vote(neighbors) -> int:
    """Most frequent label; ties go to the smallest class code."""
    counts = Counter(n.label for n in neighbors)
    return min(counts, key=lambda c: (-counts[c], c))


def heom_pair(a, b) -> float:
    """knn_base contract: one (numeric_row, nominal_row) against another, in plain Python."""
    num_a, nom_a = a
    num_b, nom_b = b
    numeric = sum(abs(float(x) - float(y)) for x, y in zip(num_a, num_b, strict=True))
    nominal = sum(x != y for x, y in zip(nom_a, nom_b, strict=True))
    return float(numeric + nominal)


def heom_matrix(train, test) -> np.ndarray:
    """knn_fast contract: (n_test, n_train) matrix. Loop over test rows, vectorize over train."""
    tr_num = np.asarray(train.X_num, dtype=float)
    te_num = np.asarray(test.X_num, dtype=float)
    tr_nom = np.asarray(train.X_nom)
    te_nom = np.asarray(test.X_nom)
    out = np.empty((len(te_nom), len(tr_nom)))
    for i in range(len(te_nom)):
        out[i] = np.abs(tr_num - te_num[i]).sum(axis=1) + (tr_nom != te_nom[i]).sum(axis=1)
    return out


def benchmark_fold(dataset: str, fold: int) -> tuple[float, float, float, int]:
    """Time the three variants on one fold; return (base, fast_per_k, fast_once, n_different)."""
    train, test, _ = preprocess_fold(dataset, fold)

    # knn_base: pairwise Python distance, everything recomputed for every K.
    start = perf_counter()
    base = {k: knn_base.knn_algorithm(train, test, k, heom_pair, majority_vote) for k in K_VALUES}
    t_base = perf_counter() - start

    # knn_fast called once per K (isolates the numpy effect).
    start = perf_counter()
    for k in K_VALUES:
        knn_fast.knn_algorithm(train, test, k, heom_matrix, majority_vote)
    t_fast_per_k = perf_counter() - start

    # knn_fast as the grid will use it: neighbours once, predict for every K.
    start = perf_counter()
    neighbors = knn_fast.nearest_neighbors(train, test, heom_matrix, k_max=max(K_VALUES))
    fast = {k: knn_fast.predict(neighbors, k, majority_vote) for k in K_VALUES}
    t_fast_once = perf_counter() - start

    n_different = sum(a != b for k in K_VALUES for a, b in zip(base[k], fast[k], strict=True))
    return t_base, t_fast_per_k, t_fast_once, n_different


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=config.DATASETS, default="splice")
    args = parser.parse_args()

    print(f"{args.dataset}, HEOM, majority vote, K={K_VALUES}, {config.N_FOLDS} folds")
    print(f"{'fold':>4} | {'knn_base':>9} | {'fast per K':>10} | {'fast once':>9} | {'diff':>4}")
    print("-" * 49)

    totals = [0.0, 0.0, 0.0]
    total_different = 0
    for fold in range(config.N_FOLDS):
        t_base, t_per_k, t_once, n_diff = benchmark_fold(args.dataset, fold)
        totals[0] += t_base
        totals[1] += t_per_k
        totals[2] += t_once
        total_different += n_diff
        print(
            f"{fold:>4} | {t_base:8.3f}s | {t_per_k:9.3f}s | {t_once:8.3f}s | {n_diff:>4}",
            flush=True,
        )

    print("-" * 49)
    sums = f"{totals[0]:8.3f}s | {totals[1]:9.3f}s | {totals[2]:8.3f}s"
    print(f"{'sum':>4} | {sums} | {total_different:>4}")
    print(
        f"speed-up vs knn_base: fast per K x{totals[0] / totals[1]:.0f}, "
        f"fast once x{totals[0] / totals[2]:.0f}"
    )
    print("diff = predictions that differ between knn_base and knn_fast (should be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
