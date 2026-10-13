import numpy as np

from experiment_runner import ClassificationResult
from models import BaseClassifier
from preprocessing import Preprocessed


class MajorityBaseline(BaseClassifier):
    def parse_params(self):
        # Baseline doesn't need parameters.
        pass

    def prepare_data(self, train: Preprocessed, test: Preprocessed):
        pass

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        majority_class = int(np.bincount(train.y_codes).argmax())
        return ClassificationResult(
            predictions=[majority_class] * len(test.y_codes),
            storage=len(train.y),
        )
