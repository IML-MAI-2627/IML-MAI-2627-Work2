import numpy as np

from experiment_runner import ClassificationResult
from models.classifier_base import BaseClassifier
from models.knn.reduction import REDUCTION_PARAM_KEYS, parse_reduction_params
from preprocessing import Preprocessed


class MajorityBaseline(BaseClassifier):
    def parse_params(self):
        # Fail-fast on 'k' parameter (non-kNN models require 'reduction_k')
        if "k" in self.params:
            raise ValueError(
                "Unknown parameter 'k' for Majority Baseline. "
                "Use 'reduction_k' to configure neighbor count for reduction."
            )

        # Baseline only accepts reduction parameters; fail-fast if unknown parameters are provided.
        unknown_params = [p for p in self.params if p not in REDUCTION_PARAM_KEYS]
        if unknown_params:
            raise ValueError(f"Unknown Baseline parameters provided: {sorted(unknown_params)}")
        self.reduction_instance = parse_reduction_params(self.params, is_knn=False)

    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> tuple[Preprocessed, Preprocessed]:
        """Validate and apply instance reduction if active."""
        if hasattr(self, "reduction_instance") and self.reduction_instance.method != "none":
            train = train.reduce(self.reduction_instance)
        return train, test

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        train, test = self.prepare_data(train, test)
        majority_class = int(np.bincount(train.y_codes).argmax())
        return ClassificationResult(
            predictions=[majority_class] * len(test.y_codes),
            storage=len(train.y),
        )
