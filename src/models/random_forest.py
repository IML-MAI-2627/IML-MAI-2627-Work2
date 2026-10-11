from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

import config
from experiment_runner import ClassificationResult
from models import BaseClassifier
from preprocessing import Preprocessed


class RandomForest(BaseClassifier):
    def parse_params(self):
        # List of parameters that must be integers if provided.
        # Other parameters are ignored.
        int_params = ["n_estimators", "max_depth", "min_samples_leaf", "random_state"]

        for param in int_params:
            if param in self.params:
                # Handle cases where the user types "--max_depth None" in the CLI
                if str(self.params[param]).lower() == "none":
                    self.params[param] = None
                else:
                    self.params[param] = int(self.params[param])

        # max_features can be int, float, "sqrt", "log2", or None
        if "max_features" in self.params:
            val = str(self.params["max_features"]).lower()
            if val == "none":
                self.params["max_features"] = None
            elif val in ["sqrt", "log2"]:
                self.params["max_features"] = val
            else:
                # If it's a number, cast to float if it has decimals, else to int
                self.params["max_features"] = float(val) if "." in val else int(val)

        if "criterion" in self.params:
            crit = str(self.params["criterion"]).lower()
            if crit in ["gini", "entropy", "log_loss"]:
                self.params["criterion"] = crit
            else:
                raise ValueError(
                    f"Invalid criterion '{crit}'. Must be one of: 'gini', 'entropy', 'log_loss'."
                )
        # Ensures reproducibility by default
        if "random_state" not in self.params:
            self.params["random_state"] = config.SEED

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
