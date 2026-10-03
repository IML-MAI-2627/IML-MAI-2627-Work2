"""Placeholders for the two most important tests of the project.

Remove the `skip` marker and implement them when the parser and preprocessing
exist (issues #3 and #4).
"""

import hashlib

import pytest

import config
from arff_parser import load_fold

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


@pytest.mark.skip(reason="Implement with the preprocessing (issue #4)")
def test_no_leakage_statistics_use_train_only():
    """Normalization ranges and imputed values come from the training fold only.

    Idea: fit on the train fold, then check that changing the test fold does not
    change the fitted statistics.
    """
