from numpy import ndarray
from sklearn.svm import SVC

import config
from experiment_runner import ClassificationResult
from models import BaseClassifier
from preprocessing import Preprocessed


class SVM(BaseClassifier):
    def parse_params(self):
        # 1. Strict Float parameters (cannot be None)
        float_params = [
            "C",
            "coef0",
            "tol",
        ]
        for param in float_params:
            if param in self.params:
                self.params[param] = float(self.params[param])

        # 2. Strict Integer parameters (cannot be None)
        int_params = ["degree", "max_iter"]
        for param in int_params:
            if param in self.params:
                self.params[param] = int(self.params[param])

        # 3. Parameters that CAN be None (e.g., random_state)
        if "random_state" in self.params:
            if str(self.params["random_state"]).lower() == "none":
                self.params["random_state"] = None
            else:
                self.params["random_state"] = int(self.params["random_state"])

        # 4. Gamma handling (can be string or float, cannot be None)
        if "gamma" in self.params:
            gamma_val = str(self.params["gamma"]).lower()
            if gamma_val in ["scale", "auto"]:
                self.params["gamma"] = gamma_val
            else:
                self.params["gamma"] = float(self.params["gamma"])

        # 5. Kernel validation (Fail-fast)
        if "kernel" in self.params:
            kernel_val = str(self.params["kernel"]).lower()
            if kernel_val in ["linear", "poly", "rbf", "sigmoid"]:
                self.params["kernel"] = kernel_val
            else:
                raise ValueError(
                    f"Invalid kernel '{kernel_val}'. Must be 'linear', 'poly', 'rbf', or 'sigmoid'."
                )

        # 6. Class weight handling
        if "class_weight" in self.params:
            cw_val = str(self.params["class_weight"]).lower()
            if cw_val == "balanced":
                self.params["class_weight"] = "balanced"
            elif cw_val == "none":
                self.params["class_weight"] = None
            else:
                raise ValueError("class_weight via CLI only supports 'balanced' or 'None'.")

        # 7. Ensure reproducibility by default
        if "random_state" not in self.params:
            self.params["random_state"] = config.SEED

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """We prepare the data for the model fit and predict."""

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
        clf = SVC(**self.params)
        train_X, train_y, test_X, test_y = self.prepare_data(train, test)
        clf.fit(train_X, train_y)

        return ClassificationResult(
            predictions=clf.predict(test_X),
            storage=len(train_y),
        )
