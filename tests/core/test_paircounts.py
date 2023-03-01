"""
NOTE: Bootstrap implementation missing.
"""
import os

import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pytest import fixture, mark, raises

from yaw.core import paircounts
from yaw.core.config import ResamplingConfig


@fixture
def tmp_hdf5():
    fpath = "_mock.hdf"
    f = h5py.File(fpath, mode="w")
    yield f
    f.close()
    os.remove(fpath)


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
    return np.diag(np.diag(patch_matrix_full))


@fixture
def patch_diag_auto(patch_matrix_auto):
    """
    These are patch totals / counts in the no-crosspatch mode for the
    autocorrelation case.
    """
    return np.diag(np.diag(patch_matrix_auto))


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
def binning():
    return pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0])


@fixture
def test_data_samples(binning):
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
        # wrong sampling method
        with raises(ValueError):
            paircounts.SampledData(binning, data, samples, method="garbage")

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


def test_binning_hdf(tmp_hdf5):
    closed = "left"
    binning = pd.IntervalIndex.from_breaks(
        np.linspace(0.0, 1.0, 11), closed=closed)
    paircounts.binning_to_hdf(binning, dest=tmp_hdf5)
    assert "binning" in tmp_hdf5
    restored = paircounts.binning_from_hdf(tmp_hdf5)
    assert restored.closed == closed
    pdt.assert_index_equal(binning, restored)


@fixture
def patched_totals_full(binning, patch_totals):
    n_bins = len(binning)
    t1, t2 = patch_totals
    return paircounts.PatchedTotal(
        binning,
        add_zbin_dimension(t1, n_bins),
        add_zbin_dimension(t2, n_bins),
        auto=False)


@fixture
def patched_totals_auto(binning, patch_totals):
    n_bins = len(binning)
    t1, t2 = patch_totals
    return paircounts.PatchedTotal(
        binning,
        add_zbin_dimension(t1, n_bins),
        add_zbin_dimension(t2, n_bins),
        auto=True)

class TestPatchedTotal:

    def test_init(self, binning, patch_totals, patched_totals_full):
        n_bins = len(binning)
        t1, t2 = patch_totals
        pt = patched_totals_full
        assert pt.n_patches == len(t1)
        assert pt.dtype == np.float_
        assert pt.ndim == 3
        assert pt.shape == (len(t1), len(t2), n_bins)
        # just call once
        repr(pt)

    def test_array(self, patched_totals_full, patch_matrix_full):
        pt = patched_totals_full
        n_bins = pt.n_bins
        # expand matrix with extra bins dimension
        full_matrix = add_zbin_dimension(patch_matrix_full, n_bins)
        # full array
        npt.assert_equal(pt.as_array(), full_matrix)
        # some slices
        npt.assert_equal(pt[:], full_matrix[:])
        npt.assert_equal(pt[-1], full_matrix[-1])
        npt.assert_equal(pt[:, -1], full_matrix[:, -1])
        npt.assert_equal(pt[:, :, -1], full_matrix[:, :, -1])
        npt.assert_equal(pt[:2], full_matrix[:2])
        npt.assert_equal(pt[:, :2], full_matrix[:, :2])
        npt.assert_equal(pt[:, :, :2], full_matrix[:, :, :2])
        npt.assert_equal(pt[1:2], full_matrix[1:2])
        npt.assert_equal(pt[:, 1:2], full_matrix[:, 1:2])
        npt.assert_equal(pt[:, :, 1:2], full_matrix[:, :, 1:2])

    def test_jackknife(
            self,
            patched_totals_full, expect_matrix_full,
            patched_totals_auto, expect_matrix_auto):
        # results for crosspatch on/off must be identical
        for crosspatch in (False, True):
            config = ResamplingConfig(method="jackknife", crosspatch=crosspatch)

            # cross-correlation case
            result = patched_totals_full.get_sum(config)
            # compare to expected results, take only first bin
            data, jack, boot = expect_matrix_full
            assert result.data[0] == data
            npt.assert_equal(result.samples[:, 0], jack)

            # autocorrelation case
            result = patched_totals_auto.get_sum(config)
            # compare to expected results, take only first bin
            data, jack, boot = expect_matrix_auto
            assert result.data[0] == data
            npt.assert_equal(result.samples[:, 0], jack)

    def test_bootstrap(self, patched_totals_full):
        with raises(NotImplementedError):
            patched_totals_full.get_sum(ResamplingConfig(method="bootstrap"))


def patched_counts_from_matrix(binning, matrix, auto):
    n_bins = len(binning)
    counts = add_zbin_dimension(matrix, n_bins)
    return paircounts.PatchedCount.from_matrix(
        binning, counts, auto=auto)


class TestPatchedCount:

    def test_init(self, binning, patch_totals, patched_totals_full):
        n_bins = len(binning)
        t1, t2 = patch_totals
        pt = patched_totals_full
        assert pt.n_patches == len(t1)
        assert pt.dtype == np.float_
        assert pt.ndim == 3
        assert pt.shape == (len(t1), len(t2), n_bins)
        # just call once
        repr(pt)

    @mark.xfail
    def test_array(self, binning, patch_matrix_full):
        counts = patched_counts_from_matrix(
            binning, patch_matrix_full, auto=False)
        n_bins = counts.n_bins
        # expand matrix with extra bins dimension
        full_matrix = add_zbin_dimension(patch_matrix_full, n_bins)
        # full array
        npt.assert_equal(counts.as_array(), full_matrix)
        # some slices
        npt.assert_equal(counts[:], full_matrix[:])
        npt.assert_equal(counts[-1], full_matrix[-1])
        npt.assert_equal(counts[:, -1], full_matrix[:, -1])
        npt.assert_equal(counts[:, :, -1], full_matrix[:, :, -1])
        npt.assert_equal(counts[:2], full_matrix[:2])
        npt.assert_equal(counts[:, :2], full_matrix[:, :2])
        npt.assert_equal(counts[:, :, :2], full_matrix[:, :, :2])
        npt.assert_equal(counts[1:2], full_matrix[1:2])
        npt.assert_equal(counts[:, 1:2], full_matrix[:, 1:2])
        npt.assert_equal(counts[:, :, 1:2], full_matrix[:, :, 1:2])

    def test_jackknife(
            self, binning,
            patch_matrix_full, expect_matrix_full,
            patch_diag_full, expect_diag_full,
            patch_matrix_auto, expect_matrix_auto,
            patch_diag_auto, expect_diag_auto):
        # case: cross-patch, cross-correlation
        config = ResamplingConfig(method="jackknife", crosspatch=True)
        counts = patched_counts_from_matrix(
            binning, patch_matrix_full, auto=False)
        result = counts.get_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_matrix_full
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: diagonal, cross-correlation
        config = ResamplingConfig(method="jackknife", crosspatch=False)
        counts = patched_counts_from_matrix(
            binning, patch_diag_full, auto=False)
        result = counts.get_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_diag_full
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: cross-patch, autocorrelation
        config = ResamplingConfig(method="jackknife", crosspatch=True)
        counts = patched_counts_from_matrix(
            binning, patch_matrix_auto, auto=True)
        result = counts.get_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_matrix_auto
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: diagonal, autocorrelation
        config = ResamplingConfig(method="jackknife", crosspatch=False)
        counts = patched_counts_from_matrix(
            binning, patch_diag_auto, auto=True)
        result = counts.get_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_diag_auto
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

    def test_bootstrap(self, binning, patch_matrix_full):
        counts = patched_counts_from_matrix(
            binning, patch_matrix_full, auto=False)
        with raises(NotImplementedError):
            counts.get_sum(ResamplingConfig(method="bootstrap"))


class TestPairCountResult:
    pass
    # test from/to_file instead of from/to_hdf