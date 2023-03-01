"""
NOTE: Bootstrap implementation missing.
"""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pytest import fixture, raises

from yaw.core import paircounts


@fixture
def patch_totals():
    """
    These are mock values for the total number of objects in catalogs 1 and 2
    with 4 patches each.
    """
    t1 = np.array([1, 2, 3, 4])
    t2 = np.array([5, 4, 3, 2])
    return t1, t2


def add_zbin_dimension(array, nbins=2):
    """
    The total/count data is usually of shape (n_patches, n_bins). This
    function can emulate this by repeating the data to add a bin dimension.
    """
    binned = np.repeat(array, nbins)
    return binned.reshape((*array.shape, nbins))


@fixture
def patch_matrix_full(patch_totals):
    """
    These are the products of the patch totals, also used as pair counts matrix.
    """
    t1, t2 = patch_totals
    return np.outer(t1, t2)


@fixture
def patch_matrix_auto(patch_totals):
    """
    These are patch totals / counts for the autocorrelation case. Diagonal
    counted only half.
    """
    t1, t2 = patch_totals
    return np.triu(np.outer(t1, t2)) - 0.5*np.diag(t1*t2)


@fixture
def patch_diag_full(patch_matrix_full):
    """
    These are patch totals / counts in the no-crosspatch mode.
    """
    return np.diag(patch_matrix_full)


@fixture
def patch_diag_auto(patch_matrix_auto):
    """
    These are patch totals / counts in the no-crosspatch mode for the
    autocorrelation case.
    """
    return np.diag(patch_matrix_auto)


@fixture
def expect_matrix_full():
    """
    The sum of pair counts and total objects to expect. Implements the acutal
    count, jackknife and a single bootstrap sample.
    """
    data = 140.0
    jackknife = np.array([81.0, 80.0, 77.0, 72.0])
    bootstrap = NotImplemented
    return data, jackknife, bootstrap


@fixture
def expect_matrix_auto():
    """
    The sum of pair counts and total objects to expect for the autcorrelation
    case. Implements the acutal count, jackknife and a single bootstrap sample.
    """
    data = 40.0
    jackknife = np.array([28.5, 22.0, 20.5, 24.0])
    bootstrap = NotImplemented
    return data, jackknife, bootstrap


@fixture
def expect_diag_full():
    """
    The sum of pair counts to expect in the no-crosspatch mode. Implements the
    acutal count, jackknife and a single bootstrap sample.
    NOTE: Total objects should be identical to matrix version of this fixture.
    """
    data = 30.0
    jackknife = np.array([25.0, 22.0, 21.0, 22.0])
    bootstrap = NotImplemented
    return data, jackknife, bootstrap


@fixture
def expect_diag_auto():
    """
    The sum of pair counts to expect in the no-crosspatch mode for the
    autocorrelation case. Implements the acutal count, jackknife and a single
    bootstrap sample.
    NOTE: Total objects should be identical to matrix version of this fixture.
    """
    data = 15.0
    jackknife = np.array([12.5, 11.0, 10.5, 11.0])
    bootstrap = NotImplemented
    return data, jackknife, bootstrap


@fixture
def test_data_samples():
    binning = pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0])
    n_bins, n_samples = 2, 7
    data = np.full(n_bins, 4)
    samples = np.repeat(np.arange(1, 8), 2).reshape((n_samples, n_bins))
    return binning, data, samples


class TestSampledData:

    def test_init(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = paircounts.SampledData(binning, data, samples, method="jackknife")
        # binning and data mismatch
        with raises(ValueError, match="unexpected *"):
            paircounts.SampledData(
                binning, data[1:], samples, method="jackknife")
        # wrong sample shape
        with raises(ValueError, match="number of bins*"):
            samples1 = samples[:, :1]
            paircounts.SampledData(binning, data, samples1, method="jackknife")
        # check some of the representation
        assert f"n_bins={samples.shape[1]}" in repr(sd)
        assert f"n_samples={samples.shape[0]}" in repr(sd)  # tests .n_samples

    def test_getters(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = paircounts.SampledData(binning, data, samples, method="jackknife")
        # test shape and index
        pdt.assert_index_equal(binning, sd.get_data().index)
        pdt.assert_index_equal(binning, sd.get_samples().index)
        assert sd.get_data().shape == data.shape
        assert sd.get_samples().shape == samples.T.shape

    def test_compatible(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = paircounts.SampledData(binning, data, samples, method="jackknife")
        # trivial case
        assert sd.is_compatible(sd)
        # wrong type
        with raises(TypeError):
            sd.is_compatible(1)
        # different binning
        binning2 = pd.IntervalIndex.from_breaks([1.0, 2.0, 3.0])
        sd2 = paircounts.SampledData(
            binning2, data, samples, method="jackknife")
        assert not sd.is_compatible(sd2)
        # different number of samples
        sd2 = paircounts.SampledData(
            binning, data, samples[:-1], method="jackknife")
        assert not sd.is_compatible(sd2)
        # different method
        sd2 = paircounts.SampledData(binning, data, samples, method="bootstrap")
        assert not sd.is_compatible(sd2)

    def test_binning(self, test_data_samples):
        # methods defined in BinnedQuantity
        binning, data, samples = test_data_samples
        sd = paircounts.SampledData(binning, data, samples, method="jackknife")
        assert sd.n_bins == len(data)
        npt.assert_equal(sd.mids, np.array([0.25, 0.75]))
        npt.assert_equal(sd.edges, np.array([0.0, 0.5, 1.0]))
        npt.assert_equal(sd.dz, np.array([0.5, 0.5]))


class TestPatchedTotal:
    pass


class TestPatchedCount:
    pass


class TestPairCountResult:
    pass
    # test from/to_file instead of from/to_hdf
