"""experiments/dedup_sensitivity.py: duplicate rows of the splice training folds."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from arff_parser import TestMatrix, TrainMatrix
from experiments import dedup_sensitivity as dedup

RESULTS = config.RESULTS_DIR / "gridsearch_v2"
WINNERS = {
    "knn": {"params": {"k": "7", "distance": "euclidean"}},
    "svm": {"params": {"kernel": "rbf"}},
    "rf": {"params": {"n_estimators": "50"}},
    "reduction_champions": {"fcnn": {"version": "1"}, "del": {}},
}


def test_model_specs_put_each_reduction_on_the_best_knn():
    specs = dedup.model_specs(WINNERS)
    assert list(specs) == ["kNN", "SVM", "RF", "kNN+fcnn", "kNN+del"]
    assert specs["kNN+fcnn"] == (
        "knn",
        {"k": "7", "distance": "euclidean", "reduction": "fcnn", "version": "1"},
    )
    assert specs["kNN+del"][1]["reduction"] == "del"


def test_twin_mask_matches_whole_rows(make_data):
    train = make_data(
        TrainMatrix, [[0.0, 1.0, "red", "S"], [1.0, 1.0, "blue", None]], ["no", "yes"]
    )
    test = make_data(
        TestMatrix,
        [[0.0, 1.0, "red", "S"], [1.0, 1.0, "blue", None], [1.0, 1.0, "blue", "L"]],
        ["yes", "yes", "yes"],
    )
    assert dedup.twin_mask(train, test).tolist() == [True, True, False]


def test_full_vs_dedup_reports_no_change_when_nothing_changes():
    rows = pd.DataFrame(
        [
            {"variant": v, "model": m, "fold": f, "accuracy": 0.8 + f / 100, "macro_f1": 0.7}
            for v in ("full", "dedup")
            for m in ("kNN", "SVM")
            for f in range(10)
        ]
    )
    out = dedup.full_vs_dedup(rows)
    assert len(out) == 4  # 2 models x 2 metrics, one Holm family
    assert (out["difference"] == 0).all() and (out["folds_changed"] == 0).all()
    assert not out["significant"].any()


needs_grid = pytest.mark.skipif(
    not (RESULTS / "winners.json").exists()
    or not all(p.exists() for p in config.fold_paths("splice", 0)),
    reason="grid v2 results or splice folds not available",
)


@needs_grid
def test_full_variant_reproduces_the_grid_on_fold_0():
    winners = json.loads((RESULTS / "winners.json").read_text())["splice"]
    key, params = dedup.model_specs(winners)["kNN"]
    official = dedup.official_accuracies(RESULTS, key, params)
    assert official is not None and len(official) == 10
    _, _, train, test, removed = dedup.load_variant(0, dedup=False)
    assert removed == 0
    out = dedup.run_model(key, params, train, test, cache=None)
    assert out["accuracy"] == pytest.approx(official[0])
    assert out["kept"] == len(train.y_codes)


@needs_grid
def test_dedup_variant_keeps_the_test_fold():
    train_raw, test_raw, train, test, removed = dedup.load_variant(0, dedup=True)
    assert removed > 0
    assert len(train.y_codes) == train_raw.n_instances - removed
    assert len(test.y_codes) == test_raw.n_instances
    assert np.array_equal(test.y, test_raw.y)
