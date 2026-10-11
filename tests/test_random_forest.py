from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


def test_parse_params_casts_integers():
    rf = RandomForest({"n_estimators": "50", "max_depth": "10", "min_samples_split": "5"})
    assert rf.params["n_estimators"] == 50
    assert rf.params["max_depth"] == 10
    assert rf.params["min_samples_split"] == 5


def test_parse_params_handles_none_gracefully():
    rf = RandomForest({"max_depth": "None", "random_state": "none"})
    assert rf.params["max_depth"] is None
    assert rf.params["random_state"] is None


def test_parse_params_ignores_unknown_or_default_params():
    # If a param is omitted, it shouldn't crash, allowing sklearn to use defaults.
    rf = RandomForest({"unknown_flag": "true"})
    assert "n_estimators" not in rf.params
    assert rf.params["unknown_flag"] == "true"


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

    # Ensure sklearn RandomForestClassifier was instantiated with our cast parameters
    mock_rf_class.assert_called_once_with(n_estimators=10, random_state=42)

    # Ensure fit and predict were called on the one-hot encoded mock arrays
    assert mock_rf_instance.fit.called
    assert mock_rf_instance.predict.called

    # Ensure the returned interface perfectly matches ClassificationResult
    assert isinstance(result, ClassificationResult)
    assert list(result.predictions) == [0, 1, 0]
    assert result.storage == 5
