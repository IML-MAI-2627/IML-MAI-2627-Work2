from __future__ import annotations

from time import perf_counter

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
        train = self.reduce_train(train)
        return train, test

    def fit(self, train: Preprocessed) -> MajorityBaseline:
        train = self.reduce_train(train)
        t0 = perf_counter()
        self.majority_class_ = int(np.bincount(train.y_codes).argmax())
        self.storage_ = len(train.y)
        self.fit_time = perf_counter() - t0
        return self

    def predict(self, test: Preprocessed) -> ClassificationResult:
        if getattr(self, "majority_class_", None) is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        t0 = perf_counter()
        predictions = [self.majority_class_] * len(test.y_codes)
        self.predict_time = perf_counter() - t0
        return ClassificationResult(
            predictions=predictions,
            storage=self.storage_,
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
        )

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        train, test = self.prepare_data(train, test)
        t_fit = perf_counter()
        majority_class = int(np.bincount(train.y_codes).argmax())
        self.fit_time = perf_counter() - t_fit

        t_pred = perf_counter()
        predictions = [majority_class] * len(test.y_codes)
        self.predict_time = perf_counter() - t_pred

        return ClassificationResult(
            predictions=predictions,
            storage=len(train.y),
            reduction_time=self.reduction_time,
            reduction_wall_time=self.reduction_wall_time,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
        )
