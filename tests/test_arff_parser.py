"""Unit tests for the scipy-based ARFF wrapper, using small inline ARFF files."""

import numpy as np
import pytest

from arff_parser import ArffData, ArffParseError, TestMatrix, TrainMatrix, load_arff

MIXED = """% a comment
@RELATION toy

@Attribute  height  REAL
@ATTRIBUTE 'eye colour' {blue, 'dark brown', green}
@attribute count INTEGER
@attribute class {a,b}

% another comment
@DATA
1.5,'dark brown',3,a
2.5,blue,?,b
?,?,5,a
"""


def write(tmp_path, text, name="t.arff"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_mixed_types_and_metadata(tmp_path):
    d = load_arff(write(tmp_path, MIXED), TrainMatrix)
    assert isinstance(d, TrainMatrix)
    assert d.relation == "toy"
    assert d.attribute_names == ("height", "eye colour", "count")
    assert d.attribute_types == ("numeric", "nominal", "numeric")
    assert d.nominal_values == {"eye colour": ("blue", "dark brown", "green")}
    assert d.class_name == "class" and d.class_values == ("a", "b")
    assert d.X.shape == (3, 3) and d.n_instances == 3 and d.n_features == 3
    assert list(d.y) == ["a", "b", "a"]
    assert all(isinstance(v, str) for v in d.y)


def test_values_and_missing_mask(tmp_path):
    d = load_arff(write(tmp_path, MIXED), ArffData)
    expected = np.array([[False, False, False], [False, False, True], [True, True, False]])
    assert np.array_equal(d.missing, expected)
    assert d.X[0, 0] == 1.5 and d.X[0, 1] == "dark brown" and d.X[1, 1] == "blue"
    assert np.isnan(d.X[2, 0]) and d.X[2, 1] is None and np.isnan(d.X[1, 2])
    num = d.numeric_matrix()
    assert num.dtype == np.float64 and num.shape == (3, 2)
    assert d.numeric_indices() == [0, 2] and d.nominal_indices() == [1]


def test_custom_class_index(tmp_path):
    text = "@relation r\n@attribute c {x,y}\n@attribute v numeric\n@data\nx,1\ny,2\n"
    d = load_arff(write(tmp_path, text), TestMatrix, class_index=0)
    assert isinstance(d, TestMatrix)
    assert d.class_name == "c" and d.attribute_names == ("v",)
    assert list(d.y) == ["x", "y"]


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty file
        "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\n",  # no rows
        "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\n1,z\n",  # undeclared
        "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\nfoo,x\n",  # bad number
        "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\n1\n",  # short row
        "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\n1,?\n",  # missing class
        "@relation r\n@attribute a numeric\n@attribute c numeric\n@data\n1,2\n",  # numeric class
        "@relation r\n@attribute a string\n@attribute c {x,y}\n@data\nhi,x\n",  # unsupported
        "@relation r\n@data\n1,2\n",  # no attributes
    ],
)
def test_malformed_input_raises(tmp_path, text):
    with pytest.raises(ArffParseError):
        load_arff(write(tmp_path, text), TrainMatrix)


def test_class_index_out_of_range(tmp_path):
    text = "@relation r\n@attribute a numeric\n@attribute c {x,y}\n@data\n1,x\n"
    with pytest.raises(ArffParseError):
        load_arff(write(tmp_path, text), TrainMatrix, class_index=5)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_arff(tmp_path / "nope.arff", TrainMatrix)


def test_quoted_value_only_in_later_rows_fails_loudly(tmp_path):
    """scipy only detects quotes if the first data row has them; we surface that as an error."""
    text = (
        "@relation r\n@attribute e {blue,'dark brown'}\n@attribute c {a,b}\n@data\n"
        "blue,a\n'dark brown',b\n"
    )
    with pytest.raises(ArffParseError):
        load_arff(write(tmp_path, text), TrainMatrix)
