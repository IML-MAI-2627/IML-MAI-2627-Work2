"""Tests for the preprocessing: synthetic ones always run, real-data ones skip without data."""

import numpy as np
import pytest

import config
from arff_parser import TestMatrix, TrainMatrix, load_fold
from models.knn.knn import KNN
from preprocessing import (
    DEFAULT_DROP,
    Preprocessor,
    preprocess_fold,
    reduce_preprocessed,
    remove_duplicate_rows,
)

TRAIN_ROWS = [
    [0.0, 5.0, "red", "S"],
    [10.0, 5.0, "blue", "L"],
    [4.0, 5.0, "blue", "S"],
    [6.0, 5.0, "green", "L"],
]
TRAIN_Y = ["no", "yes", "yes", "no"]


@pytest.fixture
def train(make_data):
    return make_data(TrainMatrix, TRAIN_ROWS, TRAIN_Y)


def _stats(prep):
    return [prep.min_.copy(), prep.max_.copy(), prep.mean_.copy(), prep.mode_.copy()]


# ---------- synthetic tests (run in CI) ----------


def test_minmax_uses_train_range(train):
    out = Preprocessor().fit_transform(train)
    assert out.num_names == ("a", "b")
    np.testing.assert_allclose(out.X_num[:, 0], [0.0, 1.0, 0.4, 0.6])


def test_constant_column_becomes_zero(train, make_data):
    prep = Preprocessor().fit(train)
    assert np.all(prep.transform(train).X_num[:, 1] == 0.0)
    # also for a test value different from the train constant
    test = make_data(TestMatrix, [[5.0, 99.0, "red", "S"]], ["no"])
    assert prep.transform(test).X_num[0, 1] == 0.0


def test_test_values_outside_train_range_are_clipped(train, make_data):
    prep = Preprocessor().fit(train)
    test = make_data(TestMatrix, [[-50.0, 5.0, "red", "S"], [500.0, 5.0, "red", "S"]], ["no", "no"])
    np.testing.assert_allclose(prep.transform(test).X_num[:, 0], [0.0, 1.0])


def test_nominal_codes_follow_header_order(train):
    out = Preprocessor().fit_transform(train)
    assert out.nom_names == ("color", "size")
    assert out.nom_cardinalities == (3, 2)
    # color header: red=0, green=1, blue=2 ; size header: S=0, L=1
    np.testing.assert_array_equal(out.X_nom, [[0, 0], [2, 1], [2, 0], [1, 1]])


def test_onehot(train):
    out = Preprocessor().fit_transform(train)
    onehot = out.to_onehot()
    assert onehot.shape == (4, 2 + 3 + 2)
    assert len(out.onehot_names()) == onehot.shape[1]
    assert out.onehot_names()[2:] == ("color=red", "color=green", "color=blue", "size=S", "size=L")
    np.testing.assert_array_equal(onehot[:, :2], out.X_num)
    np.testing.assert_array_equal(onehot[1, 2:], [0, 0, 1, 0, 1])  # blue, L
    # exactly one 1 per nominal attribute
    np.testing.assert_array_equal(onehot[:, 2:5].sum(axis=1), 1)
    np.testing.assert_array_equal(onehot[:, 5:].sum(axis=1), 1)


def test_class_is_kept_unchanged(train, make_data):
    prep = Preprocessor().fit(train)
    test = make_data(TestMatrix, [[1.0, 5.0, "red", "S"], [2.0, 5.0, "red", "L"]], ["yes", "no"])
    for data in (train, test):
        out = prep.transform(data)
        np.testing.assert_array_equal(out.y, data.y)
        assert out.class_values == data.class_values
        assert [out.class_values[c] for c in out.y_codes] == list(data.y)


def test_missing_values_are_imputed_with_train_stats(make_data):
    rows = [
        [0.0, 5.0, "blue", "S"],
        [10.0, 5.0, "blue", "L"],
        [None, 5.0, None, "L"],
        [2.0, 5.0, "red", None],
    ]
    train = make_data(TrainMatrix, rows, TRAIN_Y)
    prep = Preprocessor().fit(train)
    out = prep.transform(train)
    # a: train mean of (0, 10, 2) = 4 -> 0.4 ; color: train mode = blue (2) ; size: L (1)
    assert out.X_num[2, 0] == pytest.approx(0.4)
    assert out.X_nom[2, 0] == 2
    assert out.X_nom[3, 1] == 1

    # the test fold has other values, but the '?' still get the TRAIN mean and mode
    test = make_data(
        TestMatrix,
        [[None, 5.0, None, None], [9.0, 5.0, "red", "S"], [9.0, 5.0, "red", "S"]],
        ["no", "no", "no"],
    )
    out_test = prep.transform(test)
    assert out_test.X_num[0, 0] == pytest.approx(0.4)
    np.testing.assert_array_equal(out_test.X_nom[0], [2, 1])
    assert not np.isnan(out_test.X_num).any()


def test_statistics_ignore_missing_values(make_data):
    """A '?' must not change min/max/mean/mode: same stats as without those cells."""
    with_missing = make_data(
        TrainMatrix,
        [
            [2.0, 1.0, "green", "L"],
            [None, 3.0, None, "L"],
            [8.0, None, "green", None],
            [5.0, 2.0, "red", "S"],
        ],
        TRAIN_Y,
    )
    prep = Preprocessor().fit(with_missing)
    np.testing.assert_allclose(prep.min_, [2.0, 1.0])
    np.testing.assert_allclose(prep.max_, [8.0, 3.0])
    np.testing.assert_allclose(prep.mean_, [5.0, 2.0])
    np.testing.assert_array_equal(prep.mode_, [1, 1])  # green, L
    assert not np.isnan(np.concatenate(_stats(prep))).any()


def test_mode_tie_goes_to_first_in_alphabetical_order(make_data):
    rows = [[1.0, 1.0, "blue", "L"], [2.0, 2.0, "green", "S"]]
    prep = Preprocessor().fit(make_data(TrainMatrix, rows, ["no", "yes"]))
    np.testing.assert_array_equal(prep.mode_, [2, 1])  # blue before green, L before S


def test_numeric_column_with_all_values_missing_gives_no_nan(make_data):
    rows = [[None, 1.0, "red", "S"], [None, 2.0, "blue", "L"]]
    train = make_data(TrainMatrix, rows, ["no", "yes"])
    out = Preprocessor(max_missing=1.0).fit_transform(train)
    np.testing.assert_array_equal(out.X_num[:, 0], [0.0, 0.0])


def test_nominal_column_with_all_values_missing_raises(make_data):
    rows = [[1.0, 1.0, None, "S"], [2.0, 2.0, None, "L"]]
    train = make_data(TrainMatrix, rows, ["no", "yes"])
    with pytest.raises(ValueError, match="color"):
        Preprocessor().fit(train)
    # dropping the empty column is fine
    assert Preprocessor(drop=("color",)).fit_transform(train).nom_names == ("size",)


def test_drop_columns(train):
    out = Preprocessor(drop=("b", "size")).fit_transform(train)
    assert out.num_names == ("a",)
    assert out.nom_names == ("color",)
    assert out.X_num.shape == (4, 1)
    assert out.X_nom.shape == (4, 1)
    assert out.to_onehot().shape == (4, 1 + 3)


def test_drop_unknown_column_raises(train):
    with pytest.raises(ValueError, match="not in the data"):
        Preprocessor(drop=("nope",)).fit(train)


def test_undeclared_value_in_test_raises(train, make_data):
    """A nominal value that is not in the header is an error, never a silent code."""
    prep = Preprocessor().fit(train)
    test = make_data(TestMatrix, [[1.0, 5.0, "purple", "S"]], ["no"])
    with pytest.raises(ValueError, match="purple"):
        prep.transform(test)


def test_transform_before_fit_raises(train):
    with pytest.raises(RuntimeError):
        Preprocessor().transform(train)


def test_default_drop_for_vowel():
    assert DEFAULT_DROP["vowel"] == ("Train_or_Test", "Speaker_Number")
    assert Preprocessor.for_dataset("vowel").drop == ("Train_or_Test", "Speaker_Number")
    assert Preprocessor.for_dataset("splice").drop == ()


def test_no_leakage_stats_do_not_depend_on_test(train, make_data):
    prep = Preprocessor().fit(train)
    before = _stats(prep)

    # a test fold with very different values
    test = make_data(
        TestMatrix,
        [[-1e6, 1e6, "red", "S"], [1e6, -1e6, "red", "S"], [None, None, None, None]],
        ["no", "no", "no"],
    )
    out = prep.transform(test)
    for old, new in zip(before, _stats(prep), strict=True):
        np.testing.assert_array_equal(old, new)
    assert out.X_num.min() >= 0.0 and out.X_num.max() <= 1.0

    # fitting on train + test (the wrong way) gives different statistics
    both = make_data(TrainMatrix, TRAIN_ROWS + [list(r) for r in test.X[:2]], TRAIN_Y + ["no"] * 2)
    leaky = _stats(Preprocessor().fit(both))
    for old, new in zip(before, leaky, strict=True):
        assert not np.array_equal(old, new)


# ---------- drop_duplicate_rows (sensitivity analysis only) ----------

DUP_ROWS = TRAIN_ROWS + [
    [0.0, 5.0, "red", "S"],  # exact copy of row 0 (same class): removed
    [10.0, 5.0, "blue", "L"],  # same attributes as row 1 but another class: kept
    [None, 5.0, "red", None],  # missing cells ...
    [None, 5.0, "red", None],  # ... and its exact copy: removed
]
DUP_Y = TRAIN_Y + ["no", "no", "yes", "yes"]


def test_remove_duplicate_rows_keeps_first_and_label_conflicts(make_data):
    data = make_data(TrainMatrix, DUP_ROWS, DUP_Y)
    out, kept = remove_duplicate_rows(data)
    assert kept.tolist() == [0, 1, 2, 3, 5, 6]
    assert out.y.tolist() == [DUP_Y[i] for i in kept]
    assert isinstance(out, TrainMatrix)


def test_drop_duplicate_rows_changes_only_the_train(make_data):
    train = make_data(TrainMatrix, DUP_ROWS, DUP_Y)
    test = make_data(TestMatrix, DUP_ROWS, DUP_Y)  # the test has copies too: never touched
    prep = Preprocessor(drop_duplicate_rows=True)
    out_train = prep.fit_transform(train)
    assert prep.n_duplicates_removed_ == 2
    assert len(out_train.y) == len(DUP_ROWS) - 2
    out_test = prep.transform(test)
    assert len(out_test.y) == len(DUP_ROWS)
    # the statistics come from the deduplicated rows, with fit alone too
    assert prep.mean_[0] == pytest.approx(6.0)  # column a: 0, 10, 4, 6, 10
    assert Preprocessor().fit(train).mean_[0] == pytest.approx(5.0)  # with the copy of row 0
    assert Preprocessor(drop_duplicate_rows=True).fit(train).mean_[0] == pytest.approx(6.0)


def test_drop_duplicate_rows_is_off_by_default(make_data):
    train = make_data(TrainMatrix, DUP_ROWS, DUP_Y)
    prep = Preprocessor()
    assert len(prep.fit_transform(train).y) == len(DUP_ROWS)
    assert prep.n_duplicates_removed_ == 0


# ---------- real-data tests (skip if datasets/ is empty) ----------


@pytest.mark.parametrize("dataset", config.DATASETS)
def test_real_folds_are_preprocessed_correctly(dataset):
    pairs = [config.fold_paths(dataset, k) for k in range(config.N_FOLDS)]
    missing = [p for pair in pairs for p in pair if not p.exists()]
    if missing:
        pytest.skip(f"ARFF files not in datasets/{dataset}/ ({len(missing)} missing)")

    for fold in range(config.N_FOLDS):
        raw_train, raw_test = load_fold(dataset, fold)
        train, test, prep = preprocess_fold(dataset, fold)

        kept = [n for n in raw_train.attribute_names if n not in DEFAULT_DROP[dataset]]
        cards = [len(raw_train.nominal_values[n]) for n in kept if n in raw_train.nominal_values]
        n_num = len(kept) - len(cards)

        for raw, out in ((raw_train, train), (raw_test, test)):
            n = raw.n_instances
            # dropped columns are gone, the rest keep their order
            assert out.num_names + out.nom_names == tuple(
                [n_ for n_ in kept if n_ not in raw.nominal_values]
                + [n_ for n_ in kept if n_ in raw.nominal_values]
            )
            assert not set(DEFAULT_DROP[dataset]) & set(out.num_names + out.nom_names)

            # numeric values in [0,1], no NaN
            assert out.X_num.shape == (n, n_num)
            assert not np.isnan(out.X_num).any()
            if n_num:
                assert out.X_num.min() >= 0.0 and out.X_num.max() <= 1.0

            # nominal codes inside range(cardinality) of their column
            assert out.nom_cardinalities == tuple(cards)
            assert out.X_nom.shape == (n, len(cards))
            assert (out.X_nom >= 0).all()
            assert (out.X_nom < np.array(cards)).all()

            # one-hot width = numeric columns + sum of the cardinalities
            onehot = out.to_onehot()
            assert onehot.shape == (n, n_num + sum(cards))
            assert len(out.onehot_names()) == onehot.shape[1]
            np.testing.assert_array_equal(onehot[:, n_num:].sum(axis=1), len(cards))

            # class kept and unchanged
            np.testing.assert_array_equal(out.y, raw.y)
            assert [out.class_values[c] for c in out.y_codes] == list(raw.y)

        # train fold touches both ends of the range (unless the column is constant)
        varying = prep.max_ > prep.min_
        np.testing.assert_allclose(train.X_num[:, varying].min(axis=0), 0.0)
        np.testing.assert_allclose(train.X_num[:, varying].max(axis=0), 1.0)


def test_vowel_onehot_width():
    if not config.fold_paths("vowel", 0)[0].exists():
        pytest.skip("ARFF files not in datasets/vowel/")
    train, _, _ = preprocess_fold("vowel", 0)
    assert "Train_or_Test" not in train.nom_names
    assert "Speaker_Number" not in train.nom_names
    assert train.nom_names == ("Sex",)
    assert train.to_onehot().shape[1] == 10 + 2  # 10 numeric + Sex


def test_too_many_missing_raises_listing_all_columns(make_data):
    # columns a and b: 2 of 3 train values missing (67% > 50%)
    rows = [[None, None, "red", "S"], [None, None, "blue", "L"], [1.0, 2.0, "red", "S"]]
    train = make_data(TrainMatrix, rows, ["no", "yes", "no"])

    with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
        Preprocessor().fit(train)


def test_too_many_missing_are_dropped_with_flag(make_data):
    rows = [[None, None, "red", "S"], [None, None, "blue", "L"], [1.0, 2.0, "red", "S"]]
    train = make_data(TrainMatrix, rows, ["no", "yes", "no"])

    prep = Preprocessor(on_missing="drop").fit(train)
    out = prep.transform(train)

    assert prep.dropped_missing_ == ("a", "b")
    assert out.num_names == ()
    assert out.X_num.shape == (3, 0)
    assert out.nom_names == ("color", "size")


def test_invalid_missing_options_are_rejected():
    with pytest.raises(ValueError, match="max_missing"):
        Preprocessor(max_missing=1.5)
    with pytest.raises(ValueError, match="on_missing"):
        Preprocessor(on_missing="ignore")


def test_rows_with_unknown_class_are_removed_with_warning(train, make_data):
    prep = Preprocessor().fit(train)
    test = make_data(TestMatrix, [[1.0, 5.0, "red", "S"], [2.0, 5.0, "blue", "L"]], ["?", "yes"])

    with pytest.warns(UserWarning, match="Removed 1 rows"):
        out = prep.transform(test)

    assert out.y.tolist() == ["yes"]
    assert out.y_codes.tolist() == [1]
    assert out.X_num.shape[0] == out.X_nom.shape[0] == 1


def test_preprocessed_reduce_method(train):
    """Preprocessed.reduce() applies instance reduction directly."""
    from models.knn.reduction import DROPReduction

    prep = Preprocessor().fit_transform(train)
    reducer = DROPReduction(version=3, k=1)
    reduced = prep.reduce(reducer)
    assert reduced.X_num.shape[0] <= prep.X_num.shape[0]
    assert len(reduced.y_codes) == reduced.X_num.shape[0]


def test_reduce_preprocessed_helper(train):
    """reduce_preprocessed helper applies reducer to preprocessed data."""
    from models.knn.reduction import DELReduction

    prep = Preprocessor().fit_transform(train)
    reducer = DELReduction(k=1)
    reduced = reduce_preprocessed(prep, reducer)
    assert reduced.X_num.shape[0] <= prep.X_num.shape[0]


def _splice_fold0():
    paths = config.fold_paths("splice", 0)
    if not all(p.exists() for p in paths):
        pytest.skip("splice ARFF files not in datasets/")
    return load_fold("splice", 0)


def test_drop_duplicate_rows_false_changes_nothing_on_a_real_fold():
    train_raw, test_raw = _splice_fold0()
    default, explicit = Preprocessor(), Preprocessor(drop_duplicate_rows=False)
    a_train, b_train = default.fit_transform(train_raw), explicit.fit_transform(train_raw)
    a_test, b_test = default.transform(test_raw), explicit.transform(test_raw)
    for a, b in ((a_train, b_train), (a_test, b_test)):
        np.testing.assert_array_equal(a.X_num, b.X_num)
        np.testing.assert_array_equal(a.X_nom, b.X_nom)
        np.testing.assert_array_equal(a.y_codes, b.y_codes)
    params = {"k": "7", "distance": "euclidean", "voting": "inverse_distance"}
    first = KNN(dict(params)).fit_predict(a_train, a_test).predictions
    second = KNN(dict(params)).fit_predict(b_train, b_test).predictions
    assert list(first) == list(second)


def test_drop_duplicate_rows_on_splice_removes_only_exact_copies():
    train_raw, test_raw = _splice_fold0()
    prep = Preprocessor(drop_duplicate_rows=True)
    train, test = prep.fit_transform(train_raw), prep.transform(test_raw)
    full_test = Preprocessor().fit(train_raw).transform(test_raw)
    assert len(test.y) == test_raw.n_instances
    np.testing.assert_array_equal(test.y_codes, full_test.y_codes)
    rows = {
        tuple(r) + (c,) for r, c in zip(train_raw.X.tolist(), train_raw.y.tolist(), strict=True)
    }
    assert len(train.y) == len(rows) == train_raw.n_instances - prep.n_duplicates_removed_
    assert prep.n_duplicates_removed_ > 0
    kept = {tuple(r) + (c,) for r, c in zip(train.X_nom.tolist(), train.y.tolist(), strict=True)}
    assert len(kept) == len(train.y)  # no duplicate left
