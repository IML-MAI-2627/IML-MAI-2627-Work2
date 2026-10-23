from numpy import ndarray
from sklearn.svm import SVC

import config
from experiment_runner import ClassificationResult
from models.classifier_base import BaseClassifier, full_class_scores
from models.knn.reduction import REDUCTION_PARAM_KEYS, parse_reduction_params
from preprocessing import Preprocessed

default_params = {
    "C": 1.0,
    "kernel": "rbf",
    "degree": 3,
    "gamma": "scale",
    "coef0": 0.0,
    "tol": 0.001,
    "class_weight": None,
    "max_iter": -1,
    "random_state": config.SEED,
}


SVM_ALLOWED = set(default_params.keys()) | REDUCTION_PARAM_KEYS


class SVM(BaseClassifier):
    def parse_params(self):
        # 1. Fail-fast on 'k' parameter (non-kNN models require 'reduction_k')
        if "k" in self.params:
            raise ValueError(
                "Unknown parameter 'k' for SVM. "
                "Use 'reduction_k' to configure neighbor count for reduction."
            )

        # Fail-fast on unknown parameters
        unknown_params = [p for p in self.params if p not in SVM_ALLOWED]
        if unknown_params:
            print(f"Unknown SVM parameters provided: {unknown_params}")
            raise ValueError(f"Unknown SVM parameters provided: {unknown_params}")

        # 2. Kernel validation (Fail-fast) - Done early so we can prune later
        if "kernel" in self.params:
            kernel_val = str(self.params["kernel"]).lower()
            if kernel_val in ["linear", "poly", "rbf", "sigmoid"]:
                self.params["kernel"] = kernel_val
            else:
                raise ValueError(
                    f"Invalid kernel '{kernel_val}'. Must be 'linear', 'poly', 'rbf', or 'sigmoid'."
                )
        else:
            self.params["kernel"] = default_params["kernel"]

        # 3. Strict Float parameters (cannot be None)
        float_params = ["C", "coef0", "tol"]
        for param in float_params:
            if param in self.params:
                self.params[param] = float(self.params[param])
            else:
                self.params[param] = default_params[param]

        # 4. Strict Integer parameters (cannot be None)
        int_params = ["degree", "max_iter"]
        for param in int_params:
            if param in self.params:
                self.params[param] = int(self.params[param])
            else:
                self.params[param] = default_params[param]

        # 5. Parameters that CAN be None (e.g., random_state)
        if "random_state" in self.params:
            if str(self.params["random_state"]).lower() == "none":
                self.params["random_state"] = None
            else:
                self.params["random_state"] = int(self.params["random_state"])
        else:
            self.params["random_state"] = default_params["random_state"]

        # 6. Gamma handling (can be string or float, cannot be None)
        if "gamma" in self.params:
            gamma_val = str(self.params["gamma"]).lower()
            if gamma_val in ["scale", "auto"]:
                self.params["gamma"] = gamma_val
            else:
                self.params["gamma"] = float(self.params["gamma"])
        else:
            self.params["gamma"] = default_params["gamma"]

        # 7. Class weight handling
        if "class_weight" in self.params:
            cw_val = str(self.params["class_weight"]).lower()
            if cw_val == "balanced":
                self.params["class_weight"] = "balanced"
            elif cw_val == "none":
                self.params["class_weight"] = None
            else:
                raise ValueError("class_weight via CLI only supports 'balanced' or 'None'.")
        else:
            self.params["class_weight"] = default_params["class_weight"]

        # 8. Prune inactive parameters based on chosen kernel
        kernel = self.params["kernel"]

        if kernel == "linear":
            self.params.pop("gamma", None)
        # Degree is only used by the poly kernerl
        if kernel != "poly":
            self.params.pop("degree", None)
        # coef0 is ignored in linear kernel
        if kernel not in ["poly", "sigmoid"]:
            self.params.pop("coef0", None)

        # 9. Reduction parsing
        self.reduction_instance = parse_reduction_params(self.params, is_knn=False)
        self.sklearn_params = {k: v for k, v in self.params.items() if k in default_params}

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Validate and apply reduction if active, then prepare one-hot matrices."""
        train = self.reduce_train(train)

        # We use to_onehot, as we have decided for nominal variables
        train_X = train.to_onehot()

        if len(train_X) == 0:
            raise ValueError("Training data must not be empty")

        train_y = train.y_codes

        if len(train_X) != len(train_y):
            raise ValueError("Training features and labels must have the same number of instances")

        test_X = test.to_onehot()

        if len(test_X) == 0:
            raise ValueError("Test data must not be empty")

        test_y = test.y_codes

        if len(test_X) != len(test_y):
            raise ValueError("Test features and labels must have the same number of instances")

        return train_X, train_y, test_X, test_y

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        train_X, train_y, test_X, test_y = self.prepare_data(train, test)
        clf = SVC(**self.sklearn_params)
        clf.fit(train_X, train_y)

        scores = None
        if self.return_scores:
            raw = clf.decision_function(test_X)
            scores = full_class_scores(raw, clf.classes_, len(test.class_values), raw.min() - 1)

        return ClassificationResult(
            predictions=clf.predict(test_X),
            storage=len(train_y),
            scores=scores,
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
        )


def svmAlgorithm(train: Preprocessed, test: Preprocessed, **params: object) -> list[int]:
    """Predict one class code per test case with an SVM fitted on ``train``.

    Named as the assignment requires (``svmAlgorithm(...)``). ``params`` are the same as for
    the ``SVM`` adapter (e.g. ``kernel="rbf", C=1.0``), including the reduction options.
    """
    return [int(p) for p in SVM(dict(params)).fit_predict(train, test).predictions]
