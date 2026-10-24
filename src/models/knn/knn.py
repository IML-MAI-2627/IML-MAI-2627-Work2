"""kNN classifier adapter: runs kNN through the common BaseClassifier interface."""

from __future__ import annotations

from time import perf_counter

from experiment_runner import ClassificationResult
from models.classifier_base import BaseClassifier
from models.knn.distances import DISTANCES
from models.knn.knn_utils import (
    check_k,
    run_kNN,
)
from models.knn.reduction import parse_reduction_params
from models.knn.retention import RETENTION
from models.knn.voting import VOTE_SCORES, VOTING
from preprocessing import Preprocessed

default_params = {
    "k": 3,
    "distance": "euclidean",
    "voting": "majority",
    "retention": "never",
    "reduction": "none",
}

ALLOWED_PARAMS = set(default_params.keys()) | {
    "version",
    "threshold",
}


class KNN(BaseClassifier):
    """Own kNN; distance, voting, retention, and reduction are selected by name from registries."""

    def parse_params(self) -> None:
        """Validate and cast the CLI parameters, injecting defaults and pruning inactive keys."""
        # 0. Reject reduction-specific aliases (kNN uses its own k, distance, voting)
        for forbidden in ("reduction_k", "reduction_distance", "reduction_voting"):
            if forbidden in self.params:
                base = forbidden.replace("reduction_", "")
                raise ValueError(
                    f"Unknown parameter '{forbidden}' for kNN "
                    f"(kNN uses '{base}' for both classification and reduction)"
                )

        # 1. Fail-fast on unknown parameters
        unknown_params = [p for p in self.params if p not in ALLOWED_PARAMS]
        if unknown_params:
            raise ValueError(f"Unknown kNN parameters provided: {unknown_params}")

        # 2. k: strict positive integer (cannot be None)
        if "k" in self.params:
            try:
                k = int(str(self.params["k"]))
            except ValueError:
                raise ValueError(
                    f"Invalid k '{self.params['k']}'. Must be a positive integer."
                ) from None
            self.params["k"] = check_k(k)
        else:
            self.params["k"] = default_params["k"]

        # 3. Strategies selected by name
        for param, registry in (
            ("distance", DISTANCES),
            ("voting", VOTING),
            ("retention", RETENTION),
        ):
            if param in self.params:
                name = str(self.params[param]).lower()
                if name not in registry:
                    raise ValueError(
                        f"Invalid {param} '{name}'. Must be one of: {sorted(registry)}."
                    )
                self.params[param] = name
            else:
                self.params[param] = default_params[param]

        # 4. Reduction strategy and parameter parsing (validates reduction-specific parameters)
        self.reduction_instance = parse_reduction_params(self.params, is_knn=True)
        self.params["reduction"] = self.reduction_instance.method
        if self.reduction_instance.method in ("fcnn", "drop"):
            self.params["version"] = self.reduction_instance.version
        elif self.reduction_instance.method == "ennth":
            self.params["threshold"] = self.reduction_instance.threshold

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> tuple[Preprocessed, Preprocessed]:
        """Validate folds and apply instance reduction if active."""
        for name, data in (("Training", train), ("Test", test)):
            n = len(data.y_codes)
            if n == 0:
                raise ValueError(f"{name} data must not be empty")
            if len(data.X_num) != n or len(data.X_nom) != n:
                raise ValueError(
                    f"{name} features and labels must have the same number of instances"
                )

        train = self.reduce_train(train)

        return train, test

    def fit(self, train: Preprocessed) -> KNN:
        """Fit the kNN model (lazy learner: stores the training instances)."""
        if len(train.y_codes) == 0:
            raise ValueError("Training data must not be empty")
        if len(train.X_num) != len(train.y_codes) or len(train.X_nom) != len(train.y_codes):
            raise ValueError("Training features and labels must have the same number of instances")
        t_fit = perf_counter()
        train_reduced = self.reduce_train(train)
        self.train_ = train_reduced
        self.fit_time = perf_counter() - t_fit
        return self

    def predict(self, test: Preprocessed) -> ClassificationResult:
        """Predict labels on the test set using the stored training set."""
        if getattr(self, "train_", None) is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        if len(test.y_codes) == 0:
            raise ValueError("Test data must not be empty")
        if len(test.X_num) != len(test.y_codes) or len(test.X_nom) != len(test.y_codes):
            raise ValueError("Test features and labels must have the same number of instances")

        k = self.params["k"]
        dist_fn = DISTANCES[self.params["distance"]]
        voting_fn = VOTING[self.params["voting"]]
        retention_fn = (
            RETENTION[self.params["retention"]] if self.params["retention"] != "never" else None
        )

        t_pred = perf_counter()
        run = run_kNN(
            self.train_,
            test,
            k,
            dist_fn,
            voting_fn,
            retention_fn,
            score_function=VOTE_SCORES[self.params["voting"]] if self.return_scores else None,
        )
        self.predict_time = perf_counter() - t_pred

        return ClassificationResult(
            predictions=run.predictions,
            storage=run.case_base_size,
            scores=run.scores,
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
        )

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Predict with kNN; handles optional reduction and retention via run_kNN."""
        train, test = self.prepare_data(train, test)

        t_fit = perf_counter()
        self.train_ = train
        self.fit_time = perf_counter() - t_fit

        k = self.params["k"]
        dist_fn = DISTANCES[self.params["distance"]]
        voting_fn = VOTING[self.params["voting"]]
        retention_fn = (
            RETENTION[self.params["retention"]] if self.params["retention"] != "never" else None
        )

        t_pred = perf_counter()
        run = run_kNN(
            train,
            test,
            k,
            dist_fn,
            voting_fn,
            retention_fn,
            score_function=VOTE_SCORES[self.params["voting"]] if self.return_scores else None,
        )
        self.predict_time = perf_counter() - t_pred

        return ClassificationResult(
            predictions=run.predictions,
            storage=run.case_base_size,
            scores=run.scores,
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
        )
