from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from experiment_runner import ClassificationResult
from models.svm import SVM


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


def test_parse_params_casts_strict_floats():
    svm = SVM({"C": "1.5", "kernel": "sigmoid", "coef0": "0.5", "tol": "1e-4"})
    assert svm.params["C"] == 1.5
    assert svm.params["coef0"] == 0.5
    assert svm.params["tol"] == 0.0001


def test_parse_params_casts_strict_ints():
    svm = SVM({"kernel": "poly", "degree": "4", "max_iter": "-1"})
    assert svm.params["degree"] == 4
    assert svm.params["max_iter"] == -1


def test_parse_params_strict_types_fail_on_none():
    # Parameters that cannot be None
    with pytest.raises(ValueError):
        SVM({"C": "None"})
    with pytest.raises(ValueError):
        SVM({"kernel": "None"})
    with pytest.raises(ValueError):
        SVM({"gamma": "None"})
    with pytest.raises(ValueError):
        SVM({"degree": "None"})
    with pytest.raises(ValueError):
        SVM({"coef0": "None"})
    with pytest.raises(ValueError):
        SVM({"tol": "None"})
    with pytest.raises(ValueError):
        SVM({"max_iter": "None"})


def test_parse_params_handles_random_state():
    # Handled as None
    svm_none = SVM({"random_state": "none"})
    assert svm_none.params["random_state"] is None

    # Cast to int
    svm_int = SVM({"random_state": "42"})
    assert svm_int.params["random_state"] == 42

    # Injected by default
    svm_default = SVM({})
    assert svm_default.params["random_state"] == config.SEED


def test_parse_params_handles_gamma():
    svm_auto = SVM({"gamma": "auto"})
    assert svm_auto.params["gamma"] == "auto"

    svm_scale = SVM({"gamma": "SCALE"})
    assert svm_scale.params["gamma"] == "scale"

    svm_float = SVM({"gamma": "0.1"})
    assert svm_float.params["gamma"] == 0.1


def test_parse_params_validates_kernel():
    svm = SVM({"kernel": "POLY"})
    assert svm.params["kernel"] == "poly"

    with pytest.raises(ValueError, match="Invalid kernel 'patata'"):
        SVM({"kernel": "patata"})


def test_parse_params_validates_class_weight():
    svm_bal = SVM({"class_weight": "BALANCED"})
    assert svm_bal.params["class_weight"] == "balanced"

    svm_none = SVM({"class_weight": "none"})
    assert svm_none.params["class_weight"] is None

    with pytest.raises(ValueError, match="class_weight via CLI only supports"):
        SVM({"class_weight": "balanced_subsample"})


def test_parse_params_rejects_unknown():
    # To keep fail-fast, unknown parameters raise an error
    with pytest.raises(ValueError, match="Unknown SVM parameters provided"):
        SVM({"unknown_flag": "true"})


def test_parse_params_prunes_inactive_kernel_params():
    # linear: prunes gamma, degree, coef0
    svm_linear = SVM({"kernel": "linear", "gamma": "scale", "degree": 5, "coef0": 1.0})
    assert "gamma" not in svm_linear.params
    assert "degree" not in svm_linear.params
    assert "coef0" not in svm_linear.params

    # rbf (default): prunes degree, coef0
    svm_rbf = SVM({"kernel": "rbf", "degree": 5, "coef0": 1.0})
    assert "gamma" in svm_rbf.params
    assert "degree" not in svm_rbf.params
    assert "coef0" not in svm_rbf.params

    # sigmoid: prunes degree
    svm_sigmoid = SVM({"kernel": "sigmoid", "degree": 5, "coef0": 1.0})
    assert "gamma" in svm_sigmoid.params
    assert "coef0" in svm_sigmoid.params
    assert "degree" not in svm_sigmoid.params

    # poly: keeps everything
    svm_poly = SVM({"kernel": "poly", "gamma": "scale", "degree": 5, "coef0": 1.0})
    assert "gamma" in svm_poly.params
    assert "degree" in svm_poly.params
    assert "coef0" in svm_poly.params


def test_parse_params_injects_all_defaults():
    # If parameters are omitted, it should inject our predefined defaults
    svm = SVM({})

    # Defaults are injected
    assert svm.params["C"] == 1.0
    assert svm.params["kernel"] == "rbf"
    assert svm.params["gamma"] == "scale"
    assert svm.params["tol"] == 0.001
    assert svm.params["class_weight"] is None
    assert svm.params["max_iter"] == -1
    assert svm.params["random_state"] == config.SEED


def test_prepare_data_success(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)

    svm = SVM({})
    train_X, train_y, test_X, test_y = svm.prepare_data(train, test)

    assert train_X.shape == (5, 3)
    assert len(train_y) == 5
    assert test_X.shape == (3, 3)
    assert len(test_y) == 3


def test_prepare_data_empty_train(mock_preprocessed):
    train = mock_preprocessed(n_samples=0)
    test = mock_preprocessed(n_samples=3)

    svm = SVM({})
    with pytest.raises(ValueError, match="Training data must not be empty"):
        svm.prepare_data(train, test)


def test_prepare_data_empty_test(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=0)

    svm = SVM({})
    with pytest.raises(ValueError, match="Test data must not be empty"):
        svm.prepare_data(train, test)


def test_prepare_data_train_mismatch(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    train.y_codes = np.zeros(4)  # Mismatched labels
    test = mock_preprocessed(n_samples=3)

    svm = SVM({})
    with pytest.raises(ValueError, match="Training features and labels must have the same number"):
        svm.prepare_data(train, test)


def test_prepare_data_test_mismatch(mock_preprocessed):
    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)
    test.y_codes = np.zeros(2)  # Mismatched labels

    svm = SVM({})
    with pytest.raises(ValueError, match="Test features and labels must have the same number"):
        svm.prepare_data(train, test)


@patch("models.svm.SVC")
def test_fit_predict_execution(mock_svc_class, mock_preprocessed):
    mock_svc_instance = mock_svc_class.return_value
    mock_svc_instance.predict.return_value = np.array([0, 1, 0])

    train = mock_preprocessed(n_samples=5)
    test = mock_preprocessed(n_samples=3)

    svm = SVM({"C": 2.0, "kernel": "poly"})
    result = svm.fit_predict(train, test)

    # Ensure sklearn SVC was instantiated with our exact parameters
    # AND all the defaults explicitly injected
    mock_svc_class.assert_called_once_with(
        C=2.0,
        kernel="poly",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        class_weight=None,
        max_iter=-1,
        random_state=config.SEED,
    )

    assert mock_svc_instance.fit.called
    assert mock_svc_instance.predict.called

    assert isinstance(result, ClassificationResult)
    assert list(result.predictions) == [0, 1, 0]
    assert result.storage == 5
