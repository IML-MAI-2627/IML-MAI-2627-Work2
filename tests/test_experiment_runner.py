import csv

import numpy as np
import pytest

import experiment_runner


class FakeMatrix:
    def __init__(self, labels):
        self.y = labels
        self.y_codes = labels


def fake_preprocess_fold(dataset, fold):
    train = FakeMatrix([0, 1, 2])
    test = FakeMatrix([0, 1, 2, 0])
    return train, test, None


def always_zero_classifier(train, test):
    return experiment_runner.ClassificationResult(
        predictions=[0] * len(test.y_codes),
        storage=len(train.y),
    )


@pytest.fixture(autouse=True)
def clear_fold_cache():
    experiment_runner.load_fold.cache_clear()
    yield
    experiment_runner.load_fold.cache_clear()


def test_run_fold_metrics(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    result = experiment_runner.run_fold(
        dataset="vowel",
        fold=0,
        classifier=always_zero_classifier,
        model_name="dummy",
        params={"class": 0},
    )

    assert result.dataset == "vowel"
    assert result.fold == 0
    assert result.model == "dummy"
    assert result.params == "class=0"
    assert result.n_correct == 2
    assert result.n_incorrect == 2
    assert result.accuracy == 0.5
    assert result.time >= 0
    assert result.storage == 100.0


def test_format_params_is_order_independent():
    a = experiment_runner.format_params({"k": 3, "distance": "heom"})
    b = experiment_runner.format_params({"distance": "heom", "k": 3})
    assert a == b == "distance=heom;k=3"
    assert experiment_runner.format_params(None) == ""


def test_run_fold_rejects_wrong_prediction_count(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    def wrong_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=[0],
            storage=len(train.y),
        )

    with pytest.raises(ValueError, match="different number of predictions"):
        experiment_runner.run_fold(
            dataset="vowel",
            fold=0,
            classifier=wrong_classifier,
            model_name="dummy",
        )


def test_run_fold_accepts_numpy_predictions(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    def numpy_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=np.zeros(len(test.y_codes), dtype=np.int64),
            storage=len(train.y),
        )

    result = experiment_runner.run_fold(
        dataset="vowel",
        fold=0,
        classifier=numpy_classifier,
        model_name="dummy",
    )

    assert result.n_correct == 2
    assert result.accuracy == 0.5


@pytest.mark.parametrize("bad_prediction", [1.7, 2.0, True])
def test_run_fold_rejects_non_integer_predictions(monkeypatch, bad_prediction):
    monkeypatch.setattr(experiment_runner, "preprocess_fold", fake_preprocess_fold)

    def bad_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=[bad_prediction] * len(test.y_codes),
            storage=len(train.y),
        )

    with pytest.raises(ValueError, match="integer class codes"):
        experiment_runner.run_fold(
            dataset="vowel",
            fold=0,
            classifier=bad_classifier,
            model_name="dummy",
        )


def test_run_fold_rejects_invalid_storage(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    def invalid_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=[0] * len(test.y_codes),
            storage=10,
        )

    with pytest.raises(ValueError, match="Classifier storage"):
        experiment_runner.run_fold(
            dataset="vowel",
            fold=0,
            classifier=invalid_classifier,
            model_name="dummy",
        )


def test_run_fold_rejects_negative_storage(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    def negative_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=[0] * len(test.y_codes),
            storage=-1,
        )

    with pytest.raises(ValueError, match="Classifier storage"):
        experiment_runner.run_fold(
            dataset="vowel",
            fold=0,
            classifier=negative_classifier,
            model_name="dummy",
        )


def test_run_fold_records_reduced_storage(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    def reduced_classifier(train, test):
        return experiment_runner.ClassificationResult(
            predictions=[0] * len(test.y_codes),
            storage=2,
        )

    result = experiment_runner.run_fold(
        dataset="vowel",
        fold=0,
        classifier=reduced_classifier,
        model_name="reduced",
    )

    assert result.storage == pytest.approx(200 / 3)


def test_run_experiment_runs_all_folds(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )

    results = experiment_runner.run_experiment(
        dataset="vowel",
        classifier=always_zero_classifier,
        model_name="dummy",
    )

    assert len(results) == 10
    assert [result.fold for result in results] == list(range(10))


def test_run_experiment_resets_the_fixed_seed(monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "preprocess_fold",
        fake_preprocess_fold,
    )
    calls = []
    monkeypatch.setattr(experiment_runner.config, "set_seed", lambda: calls.append(None))

    experiment_runner.run_experiment(
        dataset="vowel",
        classifier=always_zero_classifier,
        model_name="dummy",
    )

    assert calls == [None]


def test_save_results_uses_required_csv_schema(tmp_path):
    results = [
        experiment_runner.ExperimentResult(
            dataset="vowel",
            fold=0,
            model="dummy",
            params="class=0",
            accuracy=0.5,
            n_correct=2,
            n_incorrect=2,
            time=0.01,
            storage=100.0,
            reduction_time=0.0,
            macro_f1=0.4,
            confusion_matrix="[[2,0],[2,0]]",
        )
    ]

    output = tmp_path / "results.csv"

    experiment_runner.save_results(results, output)

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0].keys() == {
        "dataset",
        "fold",
        "model",
        "params",
        "accuracy",
        "n_correct",
        "n_incorrect",
        "time",
        "storage",
        "reduction_time",
        "macro_f1",
        "confusion_matrix",
        "fit_time",
        "predict_time",
    }
    assert tuple(rows[0].keys()) == experiment_runner.CSV_COLUMNS

    assert rows[0]["dataset"] == "vowel"
    assert rows[0]["fold"] == "0"
    assert rows[0]["model"] == "dummy"
    assert rows[0]["params"] == "class=0"
    assert rows[0]["n_correct"] == "2"
    assert rows[0]["n_incorrect"] == "2"
    assert rows[0]["storage"] == "100.0"
    assert rows[0]["fit_time"] == "0.0"
    assert rows[0]["predict_time"] == "0.0"
    matrix = experiment_runner.parse_confusion_matrix(rows[0]["confusion_matrix"])
    assert matrix.tolist() == [[2, 0], [2, 0]]


def test_run_fold_confusion_matrix_and_macro_f1(monkeypatch):
    monkeypatch.setattr(experiment_runner, "preprocess_fold", fake_preprocess_fold)

    result = experiment_runner.run_fold(
        dataset="vowel", fold=0, classifier=always_zero_classifier, model_name="dummy"
    )

    # test labels [0, 1, 2, 0], always predicts 0: rows = true class, columns = predicted
    matrix = experiment_runner.parse_confusion_matrix(result.confusion_matrix)
    assert matrix.tolist() == [[2, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert matrix.sum() == result.n_correct + result.n_incorrect
    assert np.trace(matrix) == result.n_correct
    # class 0: TP=2, FP=2, FN=0 -> F1 = 4/6; classes 1 and 2 are never predicted -> 0
    assert result.macro_f1 == pytest.approx((4 / 6) / 3)


def test_macro_f1_perfect_and_ignores_absent_classes():
    matrix = experiment_runner.confusion_matrix([0, 1, 1], [0, 1, 1], n_classes=4)
    assert matrix.shape == (4, 4)
    assert experiment_runner.macro_f1(matrix) == 1.0


def test_folds_are_preprocessed_once(monkeypatch):
    calls = []

    def counting_preprocess(dataset, fold):
        calls.append(fold)
        return fake_preprocess_fold(dataset, fold)

    monkeypatch.setattr(experiment_runner, "preprocess_fold", counting_preprocess)

    for _ in range(2):
        experiment_runner.run_experiment(
            dataset="vowel",
            classifier=always_zero_classifier,
            model_name="dummy",
        )

    assert len(calls) == 10
