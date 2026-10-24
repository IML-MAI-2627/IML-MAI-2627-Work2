"""experiments/vowel_columns.py: the three column sets of vowel."""

import subprocess
import sys

import numpy as np
import pytest

import config
from experiments import vowel_columns
from models.knn.knn import KNN
from preprocessing import preprocess_fold

SCRIPT = config.PROJECT_ROOT / "experiments" / "vowel_columns.py"
PARAMS = {"k": "1", "distance": "heom", "voting": "majority", "retention": "never"}

needs_vowel = pytest.mark.skipif(
    not all(p.exists() for p in config.fold_paths("vowel", 0)), reason="ARFF files not in datasets/"
)


def test_variants_drop_only_the_intended_columns():
    a, b, c = (set(drop) for drop in vowel_columns.VARIANTS.values())
    assert a - b == {"Sex"}
    assert b - c == {"Speaker_Number"}
    assert c == {"Train_or_Test"}


@needs_vowel
def test_variant_b_is_the_current_preprocessing():
    accuracy, storage = vowel_columns.run_variants(PARAMS, folds=[0, 1])
    assert list(accuracy.columns) == list(vowel_columns.VARIANTS)
    assert ((accuracy >= 0) & (accuracy <= 1)).all().all()
    assert (storage == 100.0).all().all()  # never retain: the case base is the training set
    for fold in (0, 1):
        train, test, _ = preprocess_fold("vowel", fold)
        result = KNN(dict(PARAMS)).fit_predict(train, test)
        expected = np.mean(np.asarray(result.predictions) == test.y_codes)
        assert accuracy.loc[fold, "B: + Sex (current)"] == pytest.approx(expected)


def test_missing_winners_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="grid_search.py"):
        vowel_columns.best_vowel_knn(tmp_path)


@needs_vowel
def test_script_writes_the_report(tmp_path):
    winners = config.RESULTS_DIR / "gridsearch" / "winners.json"
    if not winners.exists():
        pytest.skip("the grid search results not available")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    text = (out / "vowel_columns.md").read_text()
    assert "Friedman" in text and "same speaker" in text
    assert (out / "vowel_columns_accuracy.csv").exists()
