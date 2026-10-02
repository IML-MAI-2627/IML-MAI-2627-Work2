"""Placeholders for the two most important tests of the project.

Remove the `skip` marker and implement them when the parser and preprocessing
exist (issues #3 and #4).
"""

import pytest


@pytest.mark.skip(reason="Implement with the ARFF parser (issue #3)")
def test_parser_reads_all_folds_unchanged():
    """Every train/test pair of splice and vowel is read; row counts match the files."""


@pytest.mark.skip(reason="Implement with the preprocessing (issue #4)")
def test_no_leakage_statistics_use_train_only():
    """Normalization ranges and imputed values come from the training fold only.

    Idea: fit on the train fold, then check that changing the test fold does not
    change the fitted statistics.
    """
