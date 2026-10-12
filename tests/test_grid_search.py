"""Tests for experiments/grid_search.py.

Includes unit tests for the parameter combination and pruning logic,
a mock test for the execution phase, and a mock test for the evaluation phase.
A full subprocess execution is avoided as it would trigger thousands of trainings.
"""

import subprocess
import sys
from unittest.mock import patch

import pytest

import config

# Dynamically import grid_search.py since it's in the experiments folder, not src
SCRIPT_PATH = config.PROJECT_ROOT / "experiments" / "grid_search.py"

# 1. Add the experiments folder to the Python path
EXPERIMENTS_DIR = config.PROJECT_ROOT / "experiments"
sys.path.append(str(EXPERIMENTS_DIR))

import grid_search  # noqa E402


# ---------------------------------------------------------
# 1. Smoke Test (CLI)
# ---------------------------------------------------------
def test_grid_search_cli_help():
    """Smoke test: ensures the script compiles and argparse handles inputs."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"{SCRIPT_PATH.name} not found.")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Manual Grid Search CV" in result.stdout
    assert "--dataset" in result.stdout
    assert "--classifier" in result.stdout


# ---------------------------------------------------------
# 2. Unit Tests: Combinations & Pruning
# ---------------------------------------------------------
def test_generate_combinations():
    """Tests if the Cartesian product is correctly generated from a grid."""
    dummy_grid = {
        "C": [1.0, 10.0],
        "kernel": ["linear", "rbf"],
    }

    combinations = list(grid_search.generate_combinations(dummy_grid))

    # 2 values for C * 2 values for kernel = 4 combinations
    assert len(combinations) == 4
    assert {"C": 1.0, "kernel": "linear"} in combinations
    assert {"C": 10.0, "kernel": "rbf"} in combinations


def test_prune_params_svm():
    """Tests if inactive SVM parameters are properly pruned based on the kernel."""
    base_params = {"C": 1.0, "kernel": "linear", "gamma": "scale", "degree": 3, "coef0": 0.5}

    # Linear: Should prune gamma, degree, coef0
    pruned_linear = grid_search.prune_params("svm", base_params)
    assert "gamma" not in pruned_linear
    assert "degree" not in pruned_linear
    assert "coef0" not in pruned_linear
    assert pruned_linear["C"] == 1.0

    # RBF: Should prune degree, coef0
    base_params["kernel"] = "rbf"
    pruned_rbf = grid_search.prune_params("svm", base_params)
    assert "gamma" in pruned_rbf
    assert "degree" not in pruned_rbf
    assert "coef0" not in pruned_rbf

    # Poly: Should keep everything
    base_params["kernel"] = "poly"
    pruned_poly = grid_search.prune_params("svm", base_params)
    assert "gamma" in pruned_poly
    assert "degree" in pruned_poly
    assert "coef0" in pruned_poly


def test_prune_params_rf():
    """Tests if Random Forest parameters are left untouched (no pruning needed)."""
    rf_params = {"n_estimators": 100, "max_depth": "None"}
    pruned_rf = grid_search.prune_params("rf", rf_params)

    assert pruned_rf == rf_params


# ---------------------------------------------------------
# 3. Unit Tests: Execution Phase (Mocked)
# ---------------------------------------------------------
@patch("grid_search.subprocess.run")
def test_run_grid_search_builds_correct_commands(mock_subprocess_run, tmp_path):
    """Mocks subprocess to ensure run.py commands are built correctly without executing."""
    # Temporarily patch the GRIDS dictionary to a tiny dummy grid
    tiny_grid = {"svm": {"C": [1.0, 2.0], "kernel": ["linear"]}, "gamma": ["auto"]}

    with patch.dict(grid_search.GRIDS, tiny_grid, clear=True):
        grid_search.run_grid_search(datasets=["vowel"], models=["svm"], out_dir=tmp_path)

    # It should have called subprocess.run 2 times (2 values for C)
    assert mock_subprocess_run.call_count == 2

    # Check the exact command built for the first execution
    first_call_args = mock_subprocess_run.call_args_list[0][0][0]  # Get the cmd list
    assert "experiments/run.py" in first_call_args
    assert "--dataset" in first_call_args
    assert "vowel" in first_call_args
    assert "--C" in first_call_args
    assert "1.0" in first_call_args

    # Check that inactive variables (like gamma) were pruned from the command
    assert "--gamma" not in first_call_args


@patch("grid_search.subprocess.run")
def test_run_grid_search_skips_existing_files(mock_subprocess_run, tmp_path):
    """Ensures that the grid search skips combinations if their CSV already exists."""
    tiny_grid = {"svm": {"C": [1.0, 2.0], "kernel": ["linear"]}}

    # Create a fake CSV file simulating that combination 0 was already completed
    (tmp_path / "vowel_svm_comb_0.csv").touch()

    with patch.dict(grid_search.GRIDS, tiny_grid, clear=True):
        grid_search.run_grid_search(datasets=["vowel"], models=["svm"], out_dir=tmp_path)

    # Since comb 0 exists, it should only call subprocess.run 1 time (for comb 1)
    assert mock_subprocess_run.call_count == 1


# ---------------------------------------------------------
# 4. Unit Tests: Evaluation Phase
# ---------------------------------------------------------
def test_evaluate_results_extracts_best_models(tmp_path):
    """Creates fake CSVs and tests if evaluate_results computes the correct optimal params."""

    # Create fake CSV results for two different configurations
    csv_header = "dataset,fold,model,params,accuracy,n_correct,n_incorrect,time,storage\n"

    # Config A: Average accuracy = 0.80
    csv_config_a = (
        csv_header + "splice,0,svm,C=1.0;kernel=linear,0.80,80,20,0.1,100\n"
        "splice,1,svm,C=1.0;kernel=linear,0.80,80,20,0.1,100\n"
    )

    # Config B: Average accuracy = 0.95 (This is the winner)
    csv_config_b = (
        csv_header + "splice,0,svm,C=10.0;kernel=rbf,0.95,95,5,0.2,100\n"
        "splice,1,svm,C=10.0;kernel=rbf,0.95,95,5,0.2,100\n"
    )

    (tmp_path / "splice_svm_comb_0.csv").write_text(csv_config_a)
    (tmp_path / "splice_svm_comb_1.csv").write_text(csv_config_b)

    # Run evaluation
    grid_search.evaluate_results(out_dir=tmp_path)

    # Check if the optimal_results.txt report was created
    report_file = tmp_path / "optimal_results.txt"
    assert report_file.exists()

    report_content = report_file.read_text()

    # Ensure it correctly identified the highest accuracy and its corresponding params
    assert "Best Accuracy:  0.9500" in report_content
    assert "Optimal Params: C=10.0;kernel=rbf" in report_content
    # Ensure the losing config is not listed as the winner
    assert "C=1.0;kernel=linear" not in report_content
