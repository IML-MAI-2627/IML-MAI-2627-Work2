"""The functions named in the assignment give the same predictions as the adapters used in
the experiments (kNNAlgorithm, svmAlgorithm, rfAlgorithm, reduction_kNNAlgorithm)."""

import pytest

import config
from experiment_runner import load_fold
from models.knn import kNNAlgorithm, reduction_kNNAlgorithm
from models.knn.distances import heom
from models.knn.knn import KNN
from models.knn.retention import always_retain
from models.knn.voting import inverse_distance
from models.random_forest import RandomForest, rfAlgorithm
from models.svm import SVM, svmAlgorithm

pytestmark = pytest.mark.skipif(
    not (config.DATASETS_DIR / "vowel").exists(), reason="vowel folds not available"
)


@pytest.fixture(scope="module")
def fold():
    train, test, _ = load_fold("vowel", 0)
    return train, test


def test_kNNAlgorithm_matches_adapter(fold):
    train, test = fold
    expected = KNN(
        {"k": 5, "distance": "heom", "voting": "inverse_distance", "retention": "always"}
    ).fit_predict(train, test)
    assert kNNAlgorithm(train, test, 5, heom, inverse_distance, always_retain) == list(
        expected.predictions
    )


@pytest.mark.parametrize(
    "reduction, options",
    [
        ("fcnn", {"version": 1}),
        ("ennth", {"threshold": 0.5}),
        ("drop", {"version": 3}),
        ("del", {}),
    ],
)
def test_reduction_kNNAlgorithm_matches_adapter(fold, reduction, options):
    train, test = fold
    expected = KNN({"k": 3, "reduction": reduction, **options}).fit_predict(train, test)
    predictions = reduction_kNNAlgorithm(
        train, test, 3, "euclidean", "majority", reduction, **options
    )
    assert predictions == list(expected.predictions)


def test_svmAlgorithm_matches_adapter(fold):
    train, test = fold
    expected = SVM({"kernel": "rbf", "C": 3.0}).fit_predict(train, test).predictions
    assert svmAlgorithm(train, test, kernel="rbf", C=3.0) == [int(p) for p in expected]


def test_rfAlgorithm_matches_adapter(fold):
    train, test = fold
    expected = RandomForest({"n_estimators": 20}).fit_predict(train, test).predictions
    assert rfAlgorithm(train, test, n_estimators=20) == [int(p) for p in expected]
