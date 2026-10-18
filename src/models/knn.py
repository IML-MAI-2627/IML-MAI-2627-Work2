"""kNN classifier adapter: runs the own kNN (``run_kNN``) through ``BaseClassifier``."""

from distances import DISTANCES
from experiment_runner import ClassificationResult
from knn_base import check_k, run_kNN
from models.classifier_base import BaseClassifier
from preprocessing import Preprocessed
from retention import RETENTION
from voting import VOTING

default_params = {
    "k": 3,
    "distance": "euclidean",
    "voting": "majority",
    "retention": "never",
}


class KNN(BaseClassifier):
    """Own kNN; distance, voting and retention are selected by name from their registries."""

    def parse_params(self) -> None:
        """Validate and cast the CLI parameters, injecting the defaults."""
        # 0. Fail-fast on unknown parameters
        unknown_params = [p for p in self.params if p not in default_params]
        if unknown_params:
            raise ValueError(f"Unknown kNN parameters provided: {unknown_params}")

        # 1. k: strict positive integer (cannot be None)
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

        # 2. Strategies are selected by name, so params stay plain strings in the CSV
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

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> tuple[Preprocessed, Preprocessed]:
        """Validate the folds; each distance chooses its own representation of them."""

        for name, data in (("Training", train), ("Test", test)):
            n = len(data.y_codes)
            if n == 0:
                raise ValueError(f"{name} data must not be empty")
            if len(data.X_num) != n or len(data.X_nom) != n:
                raise ValueError(
                    f"{name} features and labels must have the same number of instances"
                )

        # Returned unchanged: the folds are cached by the runner and must not be modified.
        return train, test

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Predict with the own kNN; storage is the case base size after the test phase."""
        train, test = self.prepare_data(train, test)
        retention = self.params["retention"]
        run = run_kNN(
            train,
            test,
            self.params["k"],
            DISTANCES[self.params["distance"]],
            VOTING[self.params["voting"]],
            # "never" uses the fast batch path (same predictions, see the tests).
            None if retention == "never" else RETENTION[retention],
        )

        return ClassificationResult(
            predictions=run.predictions,
            storage=run.case_base_size,
        )
