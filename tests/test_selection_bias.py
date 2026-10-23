"""experiments/selection_bias.py: leave-one-fold-out estimate of the selection bias."""

import pandas as pd
import pytest

from experiments.selection_bias import leave_one_fold_out


def rows(accuracy: dict[str, list[float]], time: dict[str, float] | None = None) -> pd.DataFrame:
    """Per-(config, fold) rows in the shape grid_search_v2.summarize expects."""
    time = time or {}
    records = []
    for config, accs in accuracy.items():
        for fold, acc in enumerate(accs):
            records.append(
                {
                    "config": config,
                    "key": "knn",
                    "params": config,
                    "reduction": "none",
                    "fold": fold,
                    "accuracy": acc,
                    "macro_f1": acc,
                    "time": time.get(config, 1.0),
                    "solve_time": time.get(config, 1.0),
                    "storage": 100.0,
                }
            )
    return pd.DataFrame(records)


def test_no_bias_when_one_configuration_wins_every_fold():
    result = leave_one_fold_out(rows({"good": [0.9, 0.8, 0.9, 0.8], "bad": [0.5, 0.4, 0.5, 0.4]}))
    assert result["reported"] == pytest.approx(0.85)
    assert result["bias"] == pytest.approx(0.0)
    assert result["distinct_chosen"] == 1
    assert result["rounds_with_winner"] == 4


def test_lucky_configuration_gives_a_positive_bias():
    # "lucky" wins overall only thanks to fold 0. Without fold 0, "steady" is chosen and scores
    # 0.70 on fold 0; without any other fold, "lucky" is chosen and scores 0.70 on it.
    result = leave_one_fold_out(
        rows({"lucky": [1.0, 0.70, 0.70, 0.70], "steady": [0.70, 0.75, 0.75, 0.75]})
    )
    assert result["reported"] == pytest.approx(0.775)  # mean of "lucky", the grid winner
    assert result["leave_one_fold_out"] == pytest.approx(0.70)
    assert result["bias"] == pytest.approx(0.075)
    assert result["distinct_chosen"] == 2


def test_ties_are_broken_by_time_like_the_grid():
    result = leave_one_fold_out(
        rows({"slow": [0.8, 0.8, 0.8], "fast": [0.8, 0.8, 0.8]}, time={"slow": 2.0, "fast": 1.0})
    )
    assert result["distinct_chosen"] == 1
    assert result["rounds_with_winner"] == 3
