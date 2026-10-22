"""Per-class scores (return_scores) used for ROC / PR curves, and the reduction timer."""

import numpy as np
import pytest

import config
from experiment_runner import load_fold
from models.knn.knn import KNN
from models.random_forest import RandomForest
from models.svm import SVM

pytestmark = pytest.mark.skipif(
    not (config.DATASETS_DIR / "vowel").exists(), reason="vowel folds not available"
)


@pytest.mark.parametrize(
    "model, params",
    [
        (KNN, {"k": 5, "voting": "majority"}),
        (KNN, {"k": 5, "voting": "inverse_distance", "retention": "always"}),
        (SVM, {"kernel": "rbf"}),
        (RandomForest, {"n_estimators": 20}),
    ],
)
def test_scores_shape_and_predictions_unchanged(model, params):
    train, test, _ = load_fold("vowel", 0)
    plain = model(dict(params)).fit_predict(train, test)
    assert plain.scores is None  # off by default, so timed runs do not pay for them

    scored_model = model(dict(params))
    scored_model.return_scores = True
    scored = scored_model.fit_predict(train, test)
    assert scored.scores.shape == (len(test.y_codes), len(test.class_values))
    assert list(scored.predictions) == list(plain.predictions)
    if model is KNN:  # vote shares
        np.testing.assert_allclose(scored.scores.sum(axis=1), 1.0)


def test_reduction_time_is_measured_only_with_reduction():
    train, test, _ = load_fold("vowel", 0)
    assert KNN({"k": 3}).fit_predict(train, test).reduction_time == 0.0
    reduced = KNN({"k": 3, "reduction": "fcnn"}).fit_predict(train, test)
    assert reduced.reduction_time > 0.0


def test_reduction_cache_reuses_the_reduced_set_and_its_time(tmp_path):
    from experiment_runner import run_fold
    from models.knn import reduction

    reduction.set_reduction_cache(tmp_path)
    try:
        knn = KNN({"k": 3, "reduction": "drop", "version": 3})
        first = run_fold("vowel", 0, knn.fit_predict, "kNN")
        assert len(list(tmp_path.glob("*.npz"))) == 1

        # SVM with the same reducer settings hits the cache: same instances, same time charged
        svm = SVM(
            {
                "reduction": "drop",
                "version": 3,
                "reduction_k": 3,
                "reduction_distance": "euclidean",
                "reduction_voting": "majority",
            }
        )
        second = run_fold("vowel", 0, svm.fit_predict, "SVM")
        assert len(list(tmp_path.glob("*.npz"))) == 1
        assert second.storage == first.storage
        assert second.reduction_time == first.reduction_time
        assert svm.reduction_wall_time < svm.reduction_time  # nothing recomputed
        assert second.time >= second.reduction_time  # time still includes the reduction
    finally:
        reduction.set_reduction_cache(None)
