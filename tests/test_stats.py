"""Tests for the statistical comparison module."""

import pandas as pd
import pytest

import stats


def make_rows(dataset, model, params, accuracies):
    """One runner-style row per fold with the given accuracies."""
    return [
        (dataset, fold, model, params, acc, 0, 0, 0.1, 100) for fold, acc in enumerate(accuracies)
    ]


def write_csv(path, rows):
    pd.DataFrame(rows, columns=stats.RESULT_COLUMNS).to_csv(path, index=False)
    return path


def test_load_results_joins_several_files(tmp_path):
    a = write_csv(tmp_path / "a.csv", make_rows("vowel", "kNN", "k=3", [0.9, 0.8]))
    b = write_csv(tmp_path / "b.csv", make_rows("vowel", "SVM", "", [0.7, 0.6]))

    results = stats.load_results([a, b])

    assert len(results) == 4
    assert results.loc[results["model"] == "SVM", "params"].tolist() == ["", ""]


def test_load_results_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"dataset": ["vowel"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        stats.load_results([path])


def test_config_name():
    assert stats.config_name("kNN", "k=3;vote=majority") == "kNN[k=3;vote=majority]"
    assert stats.config_name("SVM", "") == "SVM"


def test_score_table_has_folds_as_rows_and_configs_as_columns():
    rows = make_rows("vowel", "kNN", "k=3", [0.9, 0.8, 0.7])
    rows += make_rows("vowel", "SVM", "", [0.6, 0.5, 0.4])
    rows += make_rows("splice", "kNN", "k=3", [0.1, 0.1, 0.1])
    results = pd.DataFrame(rows, columns=stats.RESULT_COLUMNS)

    table = stats.score_table(results, "vowel")

    assert table.index.tolist() == [0, 1, 2]
    assert table.columns.tolist() == ["SVM", "kNN[k=3]"]
    assert table["kNN[k=3]"].tolist() == [0.9, 0.8, 0.7]


def test_score_table_rejects_missing_fold():
    rows = make_rows("vowel", "kNN", "k=3", [0.9, 0.8, 0.7])
    rows += make_rows("vowel", "SVM", "", [0.6, 0.5])
    results = pd.DataFrame(rows, columns=stats.RESULT_COLUMNS)

    with pytest.raises(ValueError, match="every fold"):
        stats.score_table(results, "vowel")


def test_score_table_rejects_duplicated_fold():
    rows = make_rows("vowel", "kNN", "k=3", [0.9, 0.8])
    rows += make_rows("vowel", "kNN", "k=3", [0.9, 0.8])
    results = pd.DataFrame(rows, columns=stats.RESULT_COLUMNS)

    with pytest.raises(ValueError, match="more than once"):
        stats.score_table(results, "vowel")


def test_score_table_rejects_unknown_metric_and_dataset():
    results = pd.DataFrame(make_rows("vowel", "kNN", "", [0.9]), columns=stats.RESULT_COLUMNS)

    with pytest.raises(ValueError, match="metric"):
        stats.score_table(results, "vowel", metric="f1")
    with pytest.raises(ValueError, match="No results for dataset"):
        stats.score_table(results, "splice")
