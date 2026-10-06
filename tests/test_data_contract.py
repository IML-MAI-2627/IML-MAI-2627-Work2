"""The two most important tests of the project: the folds are read unchanged
(issue #3) and the preprocessing uses train-only statistics (issue #4).
"""

import hashlib

import numpy as np
import pytest

import config
from arff_parser import TestMatrix, TrainMatrix, load_fold
from preprocessing import Preprocessor

EXPECTED = {
    # dataset: (n numeric, n nominal features, n classes)
    "splice": (0, 60, 3),
    "vowel": (10, 3, 11),
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("dataset", config.DATASETS)
def test_parser_reads_all_folds_unchanged(dataset):
    """Every train/test pair is read with the right shapes and the files stay unchanged."""
    pairs = [config.fold_paths(dataset, k) for k in range(config.N_FOLDS)]
    missing = [p for pair in pairs for p in pair if not p.exists()]
    if missing:
        pytest.skip(f"ARFF files not in datasets/{dataset}/ ({len(missing)} missing)")

    n_num, n_nom, n_classes = EXPECTED[dataset]
    totals = set()
    for fold, (train_path, test_path) in enumerate(pairs):
        hashes = (_sha256(train_path), _sha256(test_path))
        train, test = load_fold(dataset, fold)
        assert hashes == (_sha256(train_path), _sha256(test_path))

        for part in (train, test):
            assert part.n_features == n_num + n_nom
            assert len(part.numeric_indices()) == n_num
            assert len(part.nominal_indices()) == n_nom
            assert len(part.class_values) == n_classes
            assert part.X.shape[0] == part.y.shape[0] == part.missing.shape[0]
        assert train.attribute_names == test.attribute_names
        assert train.attribute_types == test.attribute_types
        assert train.nominal_values == test.nominal_values
        assert train.class_values == test.class_values
        totals.add(train.n_instances + test.n_instances)
    assert len(totals) == 1  # same total size in every fold


def test_no_leakage_statistics_use_train_only(make_data):
    """Normalization ranges and imputed values come from the training fold only.

    Fit on the train fold, then check that a very different test fold does not change
    the fitted statistics, and that fitting on train + test would change them.
    """
    train_rows = [
        [0.0, 1.0, "red", "S"],
        [10.0, 3.0, "blue", "L"],
        [4.0, None, "blue", "S"],
        [None, 2.0, None, "S"],
    ]
    train_y = ["no", "yes", "yes", "no"]
    test_rows = [
        [-1000.0, 1000.0, "green", "L"],
        [1000.0, -1000.0, "green", "L"],
        [500.0, 500.0, "green", "L"],
        [None, None, None, None],
    ]
    test_y = ["no", "no", "yes", "yes"]

    def stats(prep):
        return [prep.min_.copy(), prep.max_.copy(), prep.mean_.copy(), prep.mode_.copy()]

    prep = Preprocessor().fit(make_data(TrainMatrix, train_rows, train_y))
    fitted = stats(prep)
    np.testing.assert_allclose(fitted[0], [0.0, 1.0])  # min
    np.testing.assert_allclose(fitted[1], [10.0, 3.0])  # max
    np.testing.assert_allclose(fitted[2], [14 / 3, 2.0])  # mean
    np.testing.assert_array_equal(fitted[3], [2, 0])  # mode: blue, S

    # 1. transforming the test fold does not change the statistics
    out = prep.transform(make_data(TestMatrix, test_rows, test_y))
    for before, after in zip(fitted, stats(prep), strict=True):
        np.testing.assert_array_equal(before, after)

    # 2. the test fold is scaled with the train range (clipped) and imputed with train stats
    assert out.X_num.min() >= 0.0 and out.X_num.max() <= 1.0
    np.testing.assert_allclose(out.X_num[3], [(14 / 3) / 10, 0.5])
    np.testing.assert_array_equal(out.X_nom[3], [2, 0])

    # 3. fitting on train + test (the wrong way) gives different statistics
    leaky = stats(Preprocessor().fit(make_data(TrainMatrix, train_rows + test_rows, train_y * 2)))
    for before, after in zip(fitted, leaky, strict=True):
        assert not np.array_equal(before, after)
