"""Tests for the statistical comparison module."""

import math

import pandas as pd
import pytest
from scipy import stats as sp_stats

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


def demsar_table6():
    """AUC of four C4.5 variants on 14 data sets, Demšar (2006), Table 6.

    The paper prints three decimals, which makes C4.5+cf and C4.5+m+cf look tied on
    'voting' (0.975), but its ranks (2 and 3) show they were not. We nudge both values
    by 0.0001 so the computed ranks match the published ones.
    """
    data = {
        "C4.5": [
            0.763,
            0.599,
            0.954,
            0.628,
            0.882,
            0.936,
            0.661,
            0.583,
            0.775,
            1.000,
            0.940,
            0.619,
            0.972,
            0.957,
        ],
        "C4.5+m": [
            0.768,
            0.591,
            0.971,
            0.661,
            0.888,
            0.931,
            0.668,
            0.583,
            0.838,
            1.000,
            0.962,
            0.666,
            0.981,
            0.978,
        ],
        "C4.5+cf": [
            0.771,
            0.590,
            0.968,
            0.654,
            0.886,
            0.916,
            0.609,
            0.563,
            0.866,
            1.000,
            0.965,
            0.614,
            0.9751,
            0.946,
        ],
        "C4.5+m+cf": [
            0.798,
            0.569,
            0.967,
            0.657,
            0.898,
            0.931,
            0.685,
            0.625,
            0.875,
            1.000,
            0.962,
            0.669,
            0.9749,
            0.970,
        ],
    }
    return pd.DataFrame(data)


def test_rank_table_gives_rank_1_to_the_best_and_averages_ties():
    table = pd.DataFrame({"A": [0.90], "B": [0.85], "C": [0.85], "D": [0.70]})

    ranks = stats.rank_table(table)

    assert ranks.iloc[0].tolist() == [1.0, 2.5, 2.5, 4.0]


def test_rank_table_lower_is_better_for_time():
    table = pd.DataFrame({"fast": [0.1], "slow": [2.0]})

    ranks = stats.rank_table(table, higher_is_better=False)

    assert ranks.iloc[0].tolist() == [1.0, 2.0]


def test_mean_ranks_reproduce_demsar_table6():
    ranks = stats.mean_ranks(demsar_table6())

    assert ranks["C4.5"] == pytest.approx(3.143, abs=1e-3)
    assert ranks["C4.5+m"] == pytest.approx(2.000, abs=1e-3)
    assert ranks["C4.5+cf"] == pytest.approx(2.893, abs=1e-3)
    assert ranks["C4.5+m+cf"] == pytest.approx(1.964, abs=1e-3)
    assert ranks.index[0] == "C4.5+m+cf"  # best first


def test_friedman_reproduces_demsar_table6():
    result = stats.friedman_test(demsar_table6())

    assert result.n_blocks == 14
    assert result.n_configs == 4
    assert result.chi2 == pytest.approx(9.28, abs=0.01)
    assert result.iman_davenport == pytest.approx(3.69, abs=0.01)
    assert result.iman_davenport_p_value < 0.05


def test_f_critical_value_matches_demsar():
    # Demšar p. 14: "The critical value of F(3,39) for alpha = 0.05 is 2.85"
    assert sp_stats.f.ppf(0.95, 3, 39) == pytest.approx(2.85, abs=0.01)


def test_friedman_identical_configurations_give_no_evidence():
    table = pd.DataFrame({"A": [0.9, 0.8, 0.7], "B": [0.9, 0.8, 0.7], "C": [0.9, 0.8, 0.7]})

    result = stats.friedman_test(table)

    assert result.chi2 == pytest.approx(0.0)
    assert result.iman_davenport_p_value == pytest.approx(1.0)


def test_friedman_same_order_in_every_fold_gives_infinite_f():
    table = pd.DataFrame({"A": [0.9, 0.9, 0.9], "B": [0.8, 0.8, 0.8], "C": [0.7, 0.7, 0.7]})

    result = stats.friedman_test(table)

    assert result.iman_davenport == math.inf
    assert result.iman_davenport_p_value == 0.0


def test_friedman_rejects_two_configurations():
    table = pd.DataFrame({"A": [0.9, 0.8], "B": [0.7, 0.6]})

    with pytest.raises(ValueError, match="Wilcoxon"):
        stats.friedman_test(table)
