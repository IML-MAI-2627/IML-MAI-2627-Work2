"""Machine Learning models package."""

from .classifier_base import BaseClassifier
from .knn import KNN
from .majority_baseline import MajorityBaseline
from .random_forest import RandomForest
from .svm import SVM

# The __all__ variable defines exactly what is exported when someone imports from this package.
__all__ = [
    "BaseClassifier",
    "KNN",
    "MajorityBaseline",
    "RandomForest",
    "SVM",
]
