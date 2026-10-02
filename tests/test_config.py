import random

import numpy as np
import pytest

from config import DATASETS, N_FOLDS, fold_paths, set_seed


def test_fold_paths_names():
    train, test = fold_paths("splice", 3)
    assert train.name == "splice.fold.000003.train.arff"
    assert test.name == "splice.fold.000003.test.arff"
    assert train.parent.name == "splice"


def test_fold_paths_cover_all_folds():
    for dataset in DATASETS:
        for fold in range(N_FOLDS):
            train, test = fold_paths(dataset, fold)
            assert train != test


@pytest.mark.parametrize("dataset, fold", [("adult", 0), ("splice", -1), ("vowel", 10)])
def test_fold_paths_invalid(dataset, fold):
    with pytest.raises(ValueError):
        fold_paths(dataset, fold)


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    second = (random.random(), np.random.rand())
    assert first == second
