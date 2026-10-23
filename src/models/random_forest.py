from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

import config
from experiment_runner import ClassificationResult
from models.classifier_base import BaseClassifier, full_class_scores
from models.knn.reduction import REDUCTION_PARAM_KEYS, parse_reduction_params
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


RF_ALLOWED = set(default_params.keys()) | REDUCTION_PARAM_KEYS


class RandomForest(BaseClassifier):
    def parse_params(self):
        # 0. Fail-fast on 'k' parameter (non-kNN models require 'reduction_k')
        if "k" in self.params:
            raise ValueError(
                "Unknown parameter 'k' for Random Forest. "
                "Use 'reduction_k' to configure neighbor count for reduction."
            )

        # Fail-fast on unknown parameters
        unknown_params = [p for p in self.params if p not in RF_ALLOWED]
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

        # 5. Reduction parsing
        self.reduction_instance = parse_reduction_params(self.params, is_knn=False)
        self.rf_params = {k: v for k, v in self.params.items() if k in default_params}

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
        clf = RandomForestClassifier(**self.rf_params)
        clf.fit(train_X, train_y)

        scores = None
        if self.return_scores:
            raw = clf.predict_proba(test_X)
            scores = full_class_scores(raw, clf.classes_, len(test.class_values), 0.0)

        return ClassificationResult(
            predictions=clf.predict(test_X),
            storage=len(train_y),
            scores=scores,
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
        )


def rfAlgorithm(train: Preprocessed, test: Preprocessed, **params: object) -> list[int]:
    """Predict one class code per test case with a Random Forest fitted on ``train``.

    Named as the assignment requires (``rfAlgorithm(...)``). ``params`` are the same as for
    the ``RandomForest`` adapter (e.g. ``n_estimators=100, criterion="gini"``).
    """
    return [int(p) for p in RandomForest(dict(params)).fit_predict(train, test).predictions]
