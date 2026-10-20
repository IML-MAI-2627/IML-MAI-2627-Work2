"""kNN package."""

from .distances import DISTANCES
from .knn import KNN, default_params
from .knn_base import (
    Neighbor,
    Neighbors,
    check_k,
    kNNAlgorithm,
    nearest_neighbors,
    predict,
    run_kNN,
)
from .reduction import (
    REDUCTION_TECHNIQUES,
    InstanceReduction,
    ReductionResult,
    drop,
    drop1,
    drop2,
    drop3,
    drop4,
    drop5,
    eenth,
    fcnn,
    reduce_dataset,
    wilson_prob,
    wilson_th,
)
from .retention import RETENTION, always_retain, different_class, never_retain
from .voting import VOTING, inverse_distance, majority

__all__ = [
    "DISTANCES",
    "KNN",
    "default_params",
    "Neighbor",
    "Neighbors",
    "check_k",
    "kNNAlgorithm",
    "nearest_neighbors",
    "predict",
    "run_kNN",
    "REDUCTION_TECHNIQUES",
    "InstanceReduction",
    "ReductionResult",
    "drop",
    "drop1",
    "drop2",
    "drop3",
    "drop4",
    "drop5",
    "eenth",
    "fcnn",
    "reduce_dataset",
    "wilson_prob",
    "wilson_th",
    "RETENTION",
    "always_retain",
    "different_class",
    "never_retain",
    "VOTING",
    "inverse_distance",
    "majority",
]
