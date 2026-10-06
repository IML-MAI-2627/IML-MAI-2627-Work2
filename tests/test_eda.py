"""Smoke test of experiments/eda.py: runs the same command as the README."""

import subprocess
import sys

import pandas as pd
import pytest

import config

SCRIPT = config.PROJECT_ROOT / "experiments" / "eda.py"

EXPECTED_FIGURES = (
    "splice_fold0_classes.png",
    "splice_fold0_positions.png",
    "vowel_fold0_classes.png",
    "vowel_fold0_hist.png",
    "vowel_fold0_pca.png",
)
# dataset: (instances in train + test, one-hot width)
EXPECTED_CSV = {"splice": (3190, 287), "vowel": (990, 12)}


def test_eda_writes_figures_and_csv(tmp_path):
    paths = [p for d in config.DATASETS for p in config.fold_paths(d, 0)]
    if not all(p.exists() for p in paths):
        pytest.skip("ARFF files not in datasets/")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", "all", "--csv", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "majority class" in result.stdout

    # exactly the expected figures, none empty
    figures = sorted(p.name for p in tmp_path.glob("*.png"))
    assert figures == sorted(EXPECTED_FIGURES)
    assert all((tmp_path / name).stat().st_size > 0 for name in figures)

    # CSVs: one-hot columns plus the class at the end
    for dataset, (n_total, width) in EXPECTED_CSV.items():
        frames = [pd.read_csv(tmp_path / f"{dataset}_fold0_{p}.csv") for p in ("train", "test")]
        assert sum(len(frame) for frame in frames) == n_total
        for frame in frames:
            assert frame.shape[1] == width + 1
            assert frame.columns[-1] == "class"
            assert not frame.isna().any().any()
