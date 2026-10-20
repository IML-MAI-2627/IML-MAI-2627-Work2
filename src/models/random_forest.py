from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

import config
from experiment_runner import ClassificationResult
from models.classifier_base import BaseClassifier
from preprocessing import Preprocessed

default_params = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": config.SEED,
    "n_jobs": None,
    "max_features": "sqrt",
}


class RandomForest(BaseClassifier):
    def parse_params(self):
        # 0. Fail-fast on unknown parameters
        unknown_params = [p for p in self.params if p not in default_params]
        if unknown_params:
            raise ValueError(f"Unknown Random Forest parameters provided: {unknown_params}")

        # 1. Strict Integer parameters (cannot be None)
        strict_int_params = ["n_estimators", "min_samples_leaf"]
        for param in strict_int_params:
            if param in self.params:
                self.params[param] = int(self.params[param])
            else:
                self.params[param] = default_params[param]

        # 2. Nullable Integer parameters (can be None)
        nullable_int_params = ["max_depth", "random_state", "n_jobs"]
        for param in nullable_int_params:
            if param in self.params:
                if str(self.params[param]).lower() == "none":
                    self.params[param] = None
                else:
                    self.params[param] = int(self.params[param])
            else:
                self.params[param] = default_params[param]

        # 3. max_features handling
        if "max_features" in self.params:
            val = str(self.params["max_features"]).lower()
            if val == "none":
                self.params["max_features"] = None
            elif val in ["sqrt", "log2"]:
                self.params["max_features"] = val
            else:
                # If it's a number, cast to float if it has decimals, else to int
                self.params["max_features"] = float(val) if "." in val else int(val)
        else:
            self.params["max_features"] = default_params["max_features"]

        # 4. Criterion validation
        if "criterion" in self.params:
            crit = str(self.params["criterion"]).lower()
            if crit in ["gini", "entropy", "log_loss"]:
                self.params["criterion"] = crit
            else:
                raise ValueError(
                    f"Invalid criterion '{crit}'. Must be one of: 'gini', 'entropy', 'log_loss'."
                )
        else:
            self.params["criterion"] = default_params["criterion"]

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> [ndarray, ndarray, ndarray, ndarray]:
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
        clf = RandomForestClassifier(**self.params)
        train_X, train_y, test_X, test_y = self.prepare_data(train, test)
        clf.fit(train_X, train_y)

        return ClassificationResult(
            predictions=clf.predict(test_X),
            storage=len(train_y),
        )
