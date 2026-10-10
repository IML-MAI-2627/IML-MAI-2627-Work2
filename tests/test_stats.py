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


# Demšar (2006), Table 5(a): two-tailed Nemenyi critical values.
NEMENYI_Q_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}
NEMENYI_Q_010 = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
}


@pytest.mark.parametrize("k", range(2, 11))
def test_nemenyi_q_reproduces_demsar_table5a(k):
    assert stats.nemenyi_q(k, 0.05) == pytest.approx(NEMENYI_Q_005[k], abs=2e-3)
    assert stats.nemenyi_q(k, 0.10) == pytest.approx(NEMENYI_Q_010[k], abs=2e-3)


def test_critical_difference_reproduces_demsar_example():
    # Demšar p. 14: k = 4, N = 14 -> CD = 1.25 (alpha 0.05) and 1.12 (alpha 0.10)
    assert stats.critical_difference(4, 14, stats.nemenyi_q(4, 0.05)) == pytest.approx(
        1.25, abs=0.01
    )
    assert stats.critical_difference(4, 14, stats.nemenyi_q(4, 0.10)) == pytest.approx(
        1.12, abs=0.01
    )


def test_nemenyi_finds_no_difference_at_005_on_table6():
    # Demšar p. 14: "even the difference between the best and the worst performing
    # algorithm is already smaller than that" (CD = 1.25)
    result = stats.nemenyi_test(demsar_table6(), alpha=0.05)

    assert not result.significant.to_numpy().any()


def test_nemenyi_finds_two_differences_at_010_on_table6():
    # Demšar p. 14: at p = 0.10, C4.5 is significantly worse than C4.5+m and C4.5+m+cf;
    # nothing can be said about C4.5+cf.
    result = stats.nemenyi_test(demsar_table6(), alpha=0.10)
    sig = result.significant

    assert sig.loc["C4.5", "C4.5+m"]
    assert sig.loc["C4.5", "C4.5+m+cf"]
    assert not sig.loc["C4.5+cf", "C4.5"]
    assert not sig.loc["C4.5+cf", "C4.5+m"]
    assert not sig.loc["C4.5+cf", "C4.5+m+cf"]
    assert not sig.loc["C4.5+m", "C4.5+m+cf"]


# Demšar (2006), Table 5(b): two-tailed Bonferroni-Dunn critical values.
BONFERRONI_DUNN_Q_005 = {
    2: 1.960,
    3: 2.241,
    4: 2.394,
    5: 2.498,
    6: 2.576,
    7: 2.638,
    8: 2.690,
    9: 2.724,
    10: 2.773,
}
BONFERRONI_DUNN_Q_010 = {
    2: 1.645,
    3: 1.960,
    4: 2.128,
    5: 2.241,
    6: 2.326,
    7: 2.394,
    8: 2.450,
    9: 2.498,
    10: 2.539,
}


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8, 10])
def test_bonferroni_dunn_q_reproduces_demsar_table5b_at_005(k):
    assert stats.bonferroni_dunn_q(k, 0.05) == pytest.approx(BONFERRONI_DUNN_Q_005[k], abs=2e-3)


@pytest.mark.parametrize("k", range(2, 11))
def test_bonferroni_dunn_q_reproduces_demsar_table5b_at_010(k):
    assert stats.bonferroni_dunn_q(k, 0.10) == pytest.approx(BONFERRONI_DUNN_Q_010[k], abs=2e-3)


def test_bonferroni_dunn_q_table5b_typo_at_k9():
    # Table 5(b) prints 2.724 for k = 9, alpha = 0.05, but z_{1 - 0.05/16} = 2.734 and the
    # neighbouring entries all follow that formula: the printed value looks like a typo.
    assert stats.bonferroni_dunn_q(9, 0.05) == pytest.approx(2.734, abs=2e-3)


def test_bonferroni_dunn_cd_reproduces_demsar_example():
    # Demšar p. 14: q_0.05 for 4 classifiers is 2.394, so CD = 1.16
    cd = stats.critical_difference(4, 14, stats.bonferroni_dunn_q(4, 0.05))
    assert cd == pytest.approx(1.16, abs=0.01)


def test_bonferroni_dunn_reproduces_demsar_conclusion_on_table6():
    # Demšar p. 14: with C4.5 as control, C4.5+m+cf is significantly better
    # (1.179 > 1.16), C4.5+cf is not (0.250), and C4.5+m is just below the CD (1.143).
    result = stats.bonferroni_dunn_test(demsar_table6(), control="C4.5", alpha=0.05)

    assert result.significant["C4.5+m+cf"]
    assert not result.significant["C4.5+cf"]
    assert not result.significant["C4.5+m"]
    assert result.rank_differences["C4.5+m"] == pytest.approx(1.143, abs=1e-3)
    assert result.rank_differences["C4.5+m+cf"] > 0  # better than the control
    assert "C4.5" not in result.significant.index


def test_bonferroni_dunn_rejects_unknown_control():
    with pytest.raises(ValueError, match="Unknown control"):
        stats.bonferroni_dunn_test(demsar_table6(), control="SVM")


# Demšar (2006), Table 2: AUC differences C4.5+m - C4.5 on 14 data sets.
DEMSAR_TABLE2_DIFFERENCES = [
    0.005,
    -0.008,
    0.017,
    0.033,
    0.006,
    -0.005,
    0.007,
    0.000,
    0.063,
    0.000,
    0.022,
    0.047,
    0.009,
    0.021,
]


def test_wilcoxon_reproduces_demsar_table2():
    # Demšar p. 8: R+ = 93, R- = 12; with N = 14 the critical value is 21, so T = 12
    # rejects the null hypothesis at alpha = 0.05.
    r_plus, r_minus, p_value = stats.wilcoxon_signed_ranks(pd.Series(DEMSAR_TABLE2_DIFFERENCES))

    assert r_plus == pytest.approx(93)
    assert r_minus == pytest.approx(12)
    assert p_value < 0.05


def test_wilcoxon_p_value_matches_scipy_without_ties_or_zeros():
    d = [0.03, -0.01, 0.05, 0.02, 0.07, -0.04, 0.06, 0.08, 0.09, 0.10]

    _, _, p_value = stats.wilcoxon_signed_ranks(pd.Series(d))

    assert p_value == pytest.approx(sp_stats.wilcoxon(d, method="exact").pvalue)


def test_wilcoxon_ignores_one_zero_when_their_number_is_odd():
    # One zero (odd) is dropped; |0.1| gets rank 1 (positive), |-0.2| rank 2 (negative).
    r_plus, r_minus, _ = stats.wilcoxon_signed_ranks(pd.Series([0.0, 0.1, -0.2]))

    assert (r_plus, r_minus) == (1, 2)


def test_wilcoxon_all_zero_differences_give_no_evidence():
    _, _, p_value = stats.wilcoxon_signed_ranks(pd.Series([0.0, 0.0, 0.0, 0.0]))

    assert p_value == 1.0


def test_compare_two_direction_for_accuracy_and_time():
    table = pd.DataFrame({"A": [0.80, 0.70, 0.75], "B": [0.90, 0.85, 0.95]})

    accuracy = stats.compare_two(table, "A", "B")  # B more accurate
    time = stats.compare_two(table, "A", "B", higher_is_better=False)  # B slower

    assert accuracy.r_plus > accuracy.r_minus
    assert accuracy.t_statistic > 0
    assert time.r_plus < time.r_minus
    assert time.t_statistic < 0


def test_compare_two_rejects_unknown_configuration():
    table = pd.DataFrame({"A": [0.8], "B": [0.9]})

    with pytest.raises(ValueError, match="Unknown configuration"):
        stats.compare_two(table, "A", "SVM")
