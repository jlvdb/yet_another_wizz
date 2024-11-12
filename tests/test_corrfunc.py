from math import inf, nan

import numpy as np
from numpy.testing import assert_almost_equal
from pytest import mark

from yaw import corrfunc


@mark.parametrize(
    "value,width,expect",
    [
        (0.1, 3, " 0."),
        (-12.3, 5, "-12.3"),
        (0.001, 9, " 0.001000"),
        (123, 9, " 123.0000"),
        (123, 3, " 123"),
        (inf, 2, " inf"),
        (nan, 6, "   nan"),
    ],
)
def test_format_float_fixed_width(value, width, expect):
    assert corrfunc.format_float_fixed_width(value, width) == expect


@mark.parametrize("closed,braces", [("left", "[)"), ("right", "(]")])
def test_create_columns(closed, braces):
    extra_cols = list("abc")
    result = corrfunc.create_columns(extra_cols, closed)

    assert tuple(result[2:]) == tuple(extra_cols)
    assert result[0][0] + result[1][-1] == braces


@mark.parametrize("closed", ["left", "right"])
def test_read_write_header(tmp_path, closed):
    path = tmp_path / "header"
    columns = corrfunc.create_columns("abc", closed)
    description = "description"

    with path.open("w") as f:
        corrfunc.write_header(f, description, columns)
    assert corrfunc.load_header(path) == (description, columns, closed)


def test_read_write_data(tmp_path):
    path = tmp_path / "samples"
    data = np.random.normal(0, 1, size=(10))
    error = np.random.normal(0, 1, size=(10))
    edges = np.linspace(0.0, 1.0, len(data) + 1)
    closed = "left"

    corrfunc.write_data(
        path,
        "description",
        zleft=edges[:-1],
        zright=edges[1:],
        data=data,
        error=error,
        closed=closed,
    )
    edges_loaded, closed_loaded, data_loaded = corrfunc.load_data(path)
    assert_almost_equal(edges, edges_loaded)
    assert closed == closed_loaded
    assert_almost_equal(data, data_loaded)


def test_read_write_samples(tmp_path):
    path = tmp_path / "samples"
    samples = np.random.normal(0, 1, size=(20, 10))
    edges = np.linspace(0.0, 1.0, samples.shape[1] + 1)

    corrfunc.write_samples(
        path,
        "description",
        zleft=edges[:-1],
        zright=edges[1:],
        samples=samples,
        closed="left",
    )
    assert_almost_equal(samples, corrfunc.load_samples(path))


def test_write_covariance(tmp_path):
    path = tmp_path / "samples"
    cov = np.random.normal(0, 1, size=(10, 10))

    corrfunc.write_covariance(path, "description", covariance=cov)
    assert_almost_equal(cov, np.loadtxt(path, comments="#"))
