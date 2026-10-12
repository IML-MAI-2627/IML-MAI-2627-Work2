from abc import ABC, abstractmethod

from numpy import ndarray

from experiment_runner import ClassificationResult
from preprocessing import Preprocessed


class BaseClassifier(ABC):
    def __init__(self, params: dict[str, object]):
        self.params = params
        self.parse_params()  # Forces subclasses to validate the given parameters.

    @abstractmethod
    def parse_params(self):
        """Override in subclasses to cast (ex. int(k)) and validate."""
        pass

    @abstractmethod
    def prepare_data(
        self, train: Preprocessed, test: Preprocessed
    ) -> [ndarray, ndarray, ndarray, ndarray]:
        """We prepare the data for the model fit and predict."""
        pass

    @abstractmethod
    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Principal fit and predict logic."""
        pass
