from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from experiment_runner import ClassificationResult
from models.random_forest import RandomForest


@pytest.fixture
def mock_preprocessed():
    """Generates a mock Preprocessed object with configurable sizes."""

    def _make_mock(n_samples=5, n_features=3):
        mock = MagicMock()
        mock.to_onehot.return_value = np.ones((n_samples, n_features))
        mock.y_codes = np.zeros(n_samples, dtype=int)
        mock.y = np.zeros(n_samples, dtype=int)  # For storage size calculation
        return mock

    return _make_mock


def test_parse_params_casts_strict_ints():
    rf = RandomForest({"n_estimators": "50", "min_samples_leaf": "5"})
    assert rf.params["n_estimators"] == 50
    assert rf.params["min_samples_leaf"] == 5


def test_parse_params_strict_types_fail_on_none():
    # Params that cannot be None
    with pytest.raises(ValueError):
        RandomForest({"n_estimators": "None"})
    with pytest.raises(ValueError):
        RandomForest({"min_samples_leaf": "None"})


def test_parse_params_handles_nullable_ints():
    rf_none = RandomForest({"max_depth": "None", "random_state": "none", "n_jobs": "none"})
    assert rf_none.params["max_depth"] is None
    assert rf_none.params["random_state"] is None
    assert rf_none.params["n_jobs"] is None

    rf_int = RandomForest({"max_depth": "10", "random_state": "42", "n_jobs": "-1"})
    assert rf_int.params["max_depth"] == 10
    assert rf_int.params["random_state"] == 42
    assert rf_int.params["n_jobs"] == -1


def test_parse_params_injects_all_defaults():
    # If parameters are omitted, it should inject our predefined defaults
    rf = RandomForest({"unknown_flag": "true"})

    # Unknown flags are kept
    assert rf.params["unknown_flag"] == "true"

    # Defaults are injected
    assert rf.params["n_estimators"] == 100
    assert rf.params["criterion"] == "gini"
    assert rf.params["max_depth"] is None
    assert rf.params["min_samples_leaf"] == 1
    assert rf.params["random_state"] == config.SEED
    assert rf.params["n_jobs"] is None
    assert rf.params["max_features"] == "sqrt"


def test_parse_params_handles_max_features():
    # String categories
    rf_sqrt = RandomForest({"max_features": "SQRT"})
    assert rf_sqrt.params["max_features"] == "sqrt"

    rf_log2 = RandomForest({"max_features": "log2"})
    assert rf_log2.params["max_features"] == "log2"

    # Float/Int casting
    rf_float = RandomForest({"max_features": "0.5"})
    assert rf_float.params["max_features"] == 0.5

    rf_int = RandomForest({"max_features": "5"})
    assert rf_int.params["max_features"] == 5

    # None handling
    rf_none = RandomForest({"max_features": "None"})
    assert rf_none.params["max_features"] is None


def test_parse_params_validates_criterion():
    # Verify accepting valid param values in uppercase
    rf = RandomForest({"criterion": "ENTROPY"})
    assert rf.params["criterion"] == "entropy"

    # Invalid criterion raises an error
    with pytest.raises(ValueError, match="Invalid criterion 'patata'"):
        RandomForest({"criterion": "patata"})


def test_prepare_data_success(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)

    rf = RandomForest({})
    train_X, train_y, test_X, test_y = rf.prepare_data(train, test)

    assert train_X.shape == (5, 3)
    assert len(train_y) == 5
    assert test_X.shape == (3, 3)
    assert len(test_y) == 3


def test_prepare_data_empty_train(mock_preprocessed):
    train = mock_preprocessed(n_samples=0)
    test = mock_preprocessed(n_samples=3)

    rf = RandomForest({})
    with pytest.raises(ValueError, match="Training data must not be empty"):
        rf.prepare_data(train, test)


def test_prepare_data_empty_test(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=0)

    rf = RandomForest({})
    with pytest.raises(ValueError, match="Test data must not be empty"):
        rf.prepare_data(train, test)


def test_prepare_data_train_mismatch(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    train.y_codes = np.zeros(4)  # Mismatched labels
    test = mock_preprocessed(n_samples=3)

    rf = RandomForest({})
    with pytest.raises(ValueError, match="Training features and labels must have the same number"):
        rf.prepare_data(train, test)


def test_prepare_data_test_mismatch(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)
    test.y_codes = np.zeros(2)  # Mismatched labels

    rf = RandomForest({})
    with pytest.raises(ValueError, match="Test features and labels must have the same number"):
        rf.prepare_data(train, test)


@patch("models.random_forest.RandomForestClassifier")
def test_fit_predict_execution(mock_rf_class, mock_preprocessed):
    # Mock the internal sklearn model
    mock_rf_instance = mock_rf_class.return_value
    mock_rf_instance.predict.return_value = np.array([0, 1, 0])

    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)

    rf = RandomForest({"n_estimators": 10, "random_state": 42})
    result = rf.fit_predict(train, test)

    # Ensure sklearn RandomForestClassifier was instantiated with our exact parameters
    # AND all the defaults explicitly injected
    mock_rf_class.assert_called_once_with(
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=None,
        max_features="sqrt",
    )

    # Ensure fit and predict were called on the one-hot encoded mock arrays
    assert mock_rf_instance.fit.called
    assert mock_rf_instance.predict.called

    # Ensure the returned interface perfectly matches ClassificationResult
    assert isinstance(result, ClassificationResult)
    assert list(result.predictions) == [0, 1, 0]
    assert result.storage == 5
