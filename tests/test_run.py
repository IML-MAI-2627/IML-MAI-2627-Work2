"""Smoke tests for experiments/run.py: executes the models end-to-end and validates outputs."""

import csv
import subprocess
import sys

import pytest

import config

SCRIPT = config.PROJECT_ROOT / "experiments" / "run.py"


@pytest.fixture
def require_datasets():
    """Skip tests if the ARFF dataset files are not available."""
    paths = [p for d in config.DATASETS for p in config.fold_paths(d, 0)]
    if not all(p.exists() for p in paths):
        pytest.skip("ARFF files not in datasets/")


def test_run_baseline_no_extra_params(tmp_path, require_datasets):
    out_file = tmp_path / "baseline_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "baseline",
            "--out",
            str(out_file),
        ],
        capture_output=True,
        text=True,
    )

    # Assert successful execution
    assert result.returncode == 0, result.stderr
    assert "Saved 10 folds" in result.stdout

    # Assert CSV structure and content
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10  # 10 folds expected
    assert rows[0]["model"] == "Majority Baseline"
    assert rows[0]["params"] == ""  # No extra params passed
    # Ensure metrics exist and make sense
    assert 0.0 <= float(rows[0]["accuracy"]) <= 1.0
    assert float(rows[0]["storage"]) == 100.0


def test_run_random_forest_with_dynamic_params(tmp_path, require_datasets):
    out_file = tmp_path / "rf_out.csv"

    # We use a very small n_estimators so the test runs blazingly fast
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "rf",
            "--out",
            str(out_file),
            "--n_estimators",
            "2",
            "--max_depth",
            "None",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    # Assert CSV captured the dynamic parameters properly
    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    assert rows[0]["model"] == "Random Forest"

    # Check that parameters were serialized into the CSV
    params_string = rows[0]["params"]
    assert "n_estimators=2" in params_string
    assert "max_depth=None" in params_string


def test_run_svm_with_dynamic_params(tmp_path, require_datasets):
    out_file = tmp_path / "svm_out.csv"

    # We use a very small n_estimators so the test runs blazingly fast
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "svm",
            "--out",
            str(out_file),
            "--C",
            "2",
            "--kernel",
            "poly",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    # Assert CSV captured the dynamic parameters properly
    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    assert rows[0]["model"] == "Support Vector Machine"

    # Check that parameters were serialized into the CSV
    params_string = rows[0]["params"]
    assert "C=2" in params_string
    assert "kernel=poly" in params_string


def test_run_knn_with_dynamic_params(tmp_path, require_datasets):
    out_file = tmp_path / "knn_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--k",
            "5",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    assert rows[0]["model"] == "kNN"

    # Explicit and injected default parameters are both serialized
    params_string = rows[0]["params"]
    assert "k=5" in params_string
    assert "distance=euclidean" in params_string
    assert "voting=majority" in params_string
    assert float(rows[0]["storage"]) == 100.0


def test_run_knn_with_reduction_params_serialized(tmp_path, require_datasets):
    out_file = tmp_path / "knn_reduction_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--reduction",
            "drop",
            "--version",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    params_string = rows[0]["params"]
    assert "reduction=drop" in params_string
    assert "version=3" in params_string
    assert "threshold" not in params_string
    assert float(rows[0]["storage"]) < 100.0


def test_run_rejects_incompatible_reduction_params(tmp_path, require_datasets):
    out_file = tmp_path / "should_fail.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--reduction",
            "eenth",
            "--version",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unknown parameter 'version' for EENTh" in result.stderr


def test_run_knn_with_del_reduction(tmp_path, require_datasets):
    out_file = tmp_path / "knn_del_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--reduction",
            "del",
            "--k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    params_string = rows[0]["params"]
    assert "reduction=del" in params_string
    assert "version" not in params_string
    assert "threshold" not in params_string
    assert float(rows[0]["storage"]) < 100.0


def test_run_svm_with_reduction(tmp_path, require_datasets):
    out_file = tmp_path / "svm_red_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "svm",
            "--out",
            str(out_file),
            "--kernel",
            "linear",
            "--reduction",
            "drop",
            "--version",
            "3",
            "--reduction_k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    params_string = rows[0]["params"]
    assert "reduction=drop" in params_string
    assert "reduction_k=3" in params_string
    assert "version=3" in params_string
    assert float(rows[0]["storage"]) < 100.0


def test_run_rf_with_del_reduction(tmp_path, require_datasets):
    out_file = tmp_path / "rf_del_out.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "rf",
            "--out",
            str(out_file),
            "--n_estimators",
            "2",
            "--reduction",
            "del",
            "--reduction_k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    with out_file.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    params_string = rows[0]["params"]
    assert "reduction=del" in params_string
    assert "reduction_k=3" in params_string
    assert float(rows[0]["storage"]) < 100.0


def test_run_rejects_del_with_version(tmp_path, require_datasets):
    out_file = tmp_path / "should_fail_del.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--reduction",
            "del",
            "--version",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "are not allowed for DEL" in result.stderr


def test_run_knn_rejects_reduction_k(tmp_path, require_datasets):
    out_file = tmp_path / "should_fail_knn_rk.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "knn",
            "--out",
            str(out_file),
            "--reduction",
            "del",
            "--reduction_k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "reduction_k" in result.stderr


def test_run_rejects_reduction_params_without_reduction(tmp_path, require_datasets):
    out_file = tmp_path / "should_fail_no_red.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "svm",
            "--out",
            str(out_file),
            "--reduction_k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "without specifying" in result.stderr
    assert "--reduction" in result.stderr


def test_run_svm_rejects_k(tmp_path, require_datasets):
    out_file = tmp_path / "should_fail_svm_k.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "svm",
            "--out",
            str(out_file),
            "--reduction",
            "del",
            "--k",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "reduction_k" in result.stderr


def test_run_rejects_malformed_unknown_args():
    # We don't need require_datasets here because it should fail at the argparse stage
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "vowel",
            "--classifier",
            "baseline",
            "--bad_flag",
            "value1",
            "value2",
        ],
        capture_output=True,
        text=True,
    )

    # Assert it fails and gives the expected error message in stderr
    assert result.returncode != 0
    assert "Two values given for a single parameter" in result.stderr
