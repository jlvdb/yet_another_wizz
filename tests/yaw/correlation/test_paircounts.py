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

from yaw.config import ResamplingConfig
from yaw.core.math import apply_slice_ndim
from yaw.correlation import paircounts


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


def add_zbin_dimension(array, nbins):
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
    return np.triu(np.outer(t1, t2)) - 0.5 * np.diag(t1 * t2)


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
    return pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0, 1.5])


@fixture
def binning_next():
    return pd.IntervalIndex.from_breaks([1.5, 2.0, 2.5, 3.0])


def modify_totals(totals, **kwargs):
    allargs = dict(
        binning=totals.get_binning(),
        totals1=totals.totals1,
        totals2=totals.totals2,
        auto=totals.auto,
    )
    allargs.update(kwargs)
    return paircounts.PatchedTotal(**allargs)


def modify_counts(counts, **kwargs):
    allargs = dict(
        binning=counts.get_binning(),
        counts=counts.counts,
        auto=counts.auto,
    )
    allargs.update(kwargs)
    return paircounts.PatchedCount(**allargs)


def test_check_mergable_bins(patched_totals_full):
    paircounts.check_mergable([patched_totals_full, patched_totals_full], patches=False)
    new = modify_totals(patched_totals_full, auto=True)
    with raises(ValueError, match="cross-"):
        paircounts.check_mergable([patched_totals_full, new], patches=False)
    with raises(ValueError, match="patch numbers"):
        paircounts.check_mergable(
            [patched_totals_full, patched_totals_full.patches[1:]], patches=False
        )
    paircounts.check_mergable(
        [patched_totals_full, patched_totals_full.bins[1:]], patches=False
    )


def test_check_mergable_patches(patched_totals_full):
    paircounts.check_mergable([patched_totals_full, patched_totals_full], patches=True)
    new = modify_totals(patched_totals_full, auto=True)
    with raises(ValueError, match="cross-"):
        paircounts.check_mergable([patched_totals_full, new], patches=False)
    paircounts.check_mergable(
        [patched_totals_full, patched_totals_full.bins[1:]], patches=False
    )


def test_patch_idx_offset(patched_totals_full):
    npt.assert_array_almost_equal(
        paircounts.patch_idx_offset([patched_totals_full]), np.zeros(1)
    )
    result = np.array([0, patched_totals_full.n_patches])
    npt.assert_array_almost_equal(
        paircounts.patch_idx_offset([patched_totals_full, patched_totals_full]), result
    )


def test_binning_hdf(tmp_hdf5):
    closed = "left"
    binning = pd.IntervalIndex.from_breaks(np.linspace(0.0, 1.0, 11), closed=closed)
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
        auto=False,
    )


@fixture
def patched_totals_auto(binning, patch_totals):
    n_bins = len(binning)
    t1, t2 = patch_totals
    return paircounts.PatchedTotal(
        binning,
        add_zbin_dimension(t1, n_bins),
        add_zbin_dimension(t2, n_bins),
        auto=True,
    )


class TestPatchedTotal:
    def test_init(self, binning, patch_totals, patched_totals_full):
        n_bins = len(binning)
        t1, t2 = patch_totals
        pt = patched_totals_full
        assert pt.n_patches == len(t1)
        assert pt.dtype == np.float64
        assert pt.ndim == 3
        assert pt.shape == (len(t1), len(t2), n_bins)
        # wrong shape for bins
        with raises(ValueError, match="dimensional"):
            paircounts.PatchedTotal(binning, t1, t2, auto=False)
        with raises(ValueError, match="binning"):
            paircounts.PatchedTotal(
                binning,
                add_zbin_dimension(t1, 1),
                add_zbin_dimension(t2, 1),
                auto=False,
            )
        # patches don't match
        with raises(ValueError, match="patches"):
            return paircounts.PatchedTotal(
                binning,
                add_zbin_dimension(t1[1:], n_bins),
                add_zbin_dimension(t2, n_bins),
                auto=True,
            )
        # just call once
        repr(pt)

    def test_eq(self, patched_totals_full):
        assert patched_totals_full == patched_totals_full
        patched_new = paircounts.PatchedTotal(
            patched_totals_full.get_binning(),
            totals1=patched_totals_full.totals2,
            totals2=patched_totals_full.totals2,
            auto=patched_totals_full.auto,
        )
        assert patched_totals_full != patched_new
        patched_new = paircounts.PatchedTotal(
            patched_totals_full.get_binning(),
            totals1=patched_totals_full.totals1,
            totals2=patched_totals_full.totals2,
            auto=not patched_totals_full.auto,
        )
        assert patched_totals_full != patched_new
        assert patched_totals_full != 1

    def test_size(self, patched_totals_full):
        np = patched_totals_full.n_patches
        nb = patched_totals_full.n_bins
        assert patched_totals_full.size == (np * np * nb)

    def test_array(self, patched_totals_full, patch_matrix_full):
        pt = patched_totals_full
        n_bins = pt.n_bins
        # expand matrix with extra bins dimension
        full_matrix = add_zbin_dimension(patch_matrix_full, n_bins)
        # full array
        npt.assert_equal(pt.as_array(), full_matrix)

    def test_jackknife(
        self,
        patched_totals_full,
        expect_matrix_full,
        patched_totals_auto,
        expect_matrix_auto,
    ):
        # results for crosspatch on/off must be identical
        for crosspatch in (False, True):
            config = ResamplingConfig(method="jackknife", crosspatch=crosspatch)

            # cross-correlation case
            result = patched_totals_full.sample_sum(config)
            # compare to expected results, take only first bin
            data, jack, boot = expect_matrix_full
            assert result.data[0] == data
            npt.assert_equal(result.samples[:, 0], jack)

            # autocorrelation case
            result = patched_totals_auto.sample_sum(config)
            # compare to expected results, take only first bin
            data, jack, boot = expect_matrix_auto
            assert result.data[0] == data
            npt.assert_equal(result.samples[:, 0], jack)

        subset = patched_totals_full.patches[0]
        assert subset.sample_sum(config).samples.ndim == 2

    def test_bootstrap(self, patched_totals_full):
        with raises(NotImplementedError):
            patched_totals_full.sample_sum(ResamplingConfig(method="bootstrap"))

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_bins(self, patched_totals_full, patch_totals, items):
        t1 = add_zbin_dimension(patch_totals[0], patched_totals_full.n_bins)
        t2 = add_zbin_dimension(patch_totals[1], patched_totals_full.n_bins)
        subset = patched_totals_full.bins[items]
        npt.assert_array_equal(subset.totals1, apply_slice_ndim(t1, items, axis=1))
        npt.assert_array_equal(subset.totals2, apply_slice_ndim(t2, items, axis=1))
        assert (subset.get_binning() == patched_totals_full.get_binning()[items]).all()

    def test_bins_iter(self, patched_totals_full, patch_totals):
        t1 = add_zbin_dimension(patch_totals[0], patched_totals_full.n_bins)
        t2 = add_zbin_dimension(patch_totals[1], patched_totals_full.n_bins)
        for i, _bin in enumerate(patched_totals_full.bins):
            npt.assert_array_equal(_bin.totals1, apply_slice_ndim(t1, i, axis=1))
            npt.assert_array_equal(_bin.totals2, apply_slice_ndim(t2, i, axis=1))

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_patches(self, patched_totals_full, patch_totals, items):
        t1 = add_zbin_dimension(patch_totals[0], patched_totals_full.n_bins)
        t2 = add_zbin_dimension(patch_totals[1], patched_totals_full.n_bins)
        subset = patched_totals_full.patches[items]
        npt.assert_array_equal(subset.totals1, apply_slice_ndim(t1, items, axis=0))
        npt.assert_array_equal(subset.totals2, apply_slice_ndim(t2, items, axis=0))
        assert (subset.get_binning() == patched_totals_full.get_binning()).all()

    def test_patches_iter(self, patched_totals_full, patch_totals):
        t1 = add_zbin_dimension(patch_totals[0], patched_totals_full.n_bins)
        t2 = add_zbin_dimension(patch_totals[1], patched_totals_full.n_bins)
        for i, _bin in enumerate(patched_totals_full.patches):
            npt.assert_array_equal(_bin.totals1, apply_slice_ndim(t1, i, axis=0))
            npt.assert_array_equal(_bin.totals2, apply_slice_ndim(t2, i, axis=0))

    def test_concatenate_patches(self, patched_totals_full, binning_next):
        merged = patched_totals_full.concatenate_patches(patched_totals_full)
        diag = np.diag(patched_totals_full.as_array()[:, :, 0])
        npt.assert_array_equal(
            np.diag(merged.as_array()[:, :, 0]), np.concatenate([diag, diag])
        )
        mod = modify_totals(patched_totals_full, binning=binning_next)
        with raises(ValueError):
            patched_totals_full.concatenate_patches(mod)

    def test_concatenate_bins(self, patched_totals_full, binning_next):
        appended = modify_totals(patched_totals_full, binning=binning_next)
        merged = patched_totals_full.concatenate_bins(appended)
        binned = patched_totals_full.as_array()[0, 0]
        npt.assert_array_equal(
            merged.as_array()[0, 0], np.concatenate([binned, binned])
        )
        with raises(ValueError):
            patched_totals_full.concatenate_bins(patched_totals_full)


def patched_counts_from_matrix(binning, matrix, auto):
    n_bins = len(binning)
    counts = add_zbin_dimension(matrix, n_bins)
    return paircounts.PatchedCount(binning, counts, auto=auto)


@fixture
def patched_counts_full(binning, patch_matrix_full):
    n_bins = len(binning)
    matrix = add_zbin_dimension(patch_matrix_full, n_bins)
    return paircounts.PatchedCount(binning, matrix, auto=False)


class TestPatchedCount:
    def test_init(self, binning, patch_matrix_full):
        n_bins = len(binning)
        matrix = add_zbin_dimension(patch_matrix_full, n_bins)
        counts = paircounts.PatchedCount(binning, matrix, auto=False)
        # wrong shape
        with raises(ValueError):
            counts = paircounts.PatchedCount(
                binning, matrix[:, :, :-1], auto=False
            )  # missing bin
        with raises(IndexError):
            counts = paircounts.PatchedCount(
                binning, matrix[:, :-1], auto=False
            )  # not square
        with raises(IndexError):
            counts = paircounts.PatchedCount(
                binning, matrix[:-1], auto=False
            )  # not square
        # just call once
        repr(counts)

    def test_eq(self, patched_counts_full):
        assert patched_counts_full == patched_counts_full
        patched_new = paircounts.PatchedCount(
            patched_counts_full.get_binning(),
            patched_counts_full.counts + 1,
            auto=patched_counts_full.auto,
        )
        assert patched_counts_full != patched_new
        patched_new = paircounts.PatchedCount(
            patched_counts_full.get_binning(),
            patched_counts_full.counts,
            auto=not patched_counts_full.auto,
        )
        assert patched_counts_full != patched_new
        assert patched_counts_full != 1

    def test_size(self, patched_counts_full):
        np = patched_counts_full.n_patches
        nb = patched_counts_full.n_bins
        assert patched_counts_full.size == (np * np * nb)

    def test_keys_values(self, binning):
        n_bins = len(binning)
        counts = paircounts.PatchedCount.zeros(binning, 2, auto=False)
        # check zero matrix
        npt.assert_equal(counts.keys(), np.empty((0, 2), dtype=np.int64))
        npt.assert_equal(counts.values(), np.empty((0, n_bins)))
        # insert single item
        key = (0, 1)
        value = [1.0] * n_bins
        counts.set_measurement(key, value)
        npt.assert_equal(counts.keys(), np.atleast_2d(key))
        npt.assert_equal(counts.values(), np.atleast_2d(value))
        # check wrong assignments
        with raises(ValueError):  # extra element
            counts.set_measurement(key, [1.0] * (n_bins + 1))
        with raises(TypeError):  # wrong key type
            counts.set_measurement(1, [1.0] * n_bins)
        with raises(IndexError):  # wrong key shape
            counts.set_measurement((1,), [1.0] * n_bins)

    def test_array(self, binning, patch_matrix_full):
        counts = patched_counts_from_matrix(binning, patch_matrix_full, auto=False)
        n_bins = counts.n_bins
        # expand matrix with extra bins dimension
        full_matrix = add_zbin_dimension(patch_matrix_full, n_bins)
        # full array
        npt.assert_equal(counts.as_array(), full_matrix)

    def test_jackknife(
        self,
        binning,
        patch_matrix_full,
        expect_matrix_full,
        patch_diag_full,
        expect_diag_full,
        patch_matrix_auto,
        expect_matrix_auto,
        patch_diag_auto,
        expect_diag_auto,
    ):
        # case: cross-patch, cross-correlation
        config = ResamplingConfig(method="jackknife", crosspatch=True)
        counts = patched_counts_from_matrix(binning, patch_matrix_full, auto=False)
        result = counts.sample_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_matrix_full
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: diagonal, cross-correlation
        config = ResamplingConfig(method="jackknife", crosspatch=False)
        counts = patched_counts_from_matrix(binning, patch_diag_full, auto=False)
        result = counts.sample_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_diag_full
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: cross-patch, autocorrelation
        config = ResamplingConfig(method="jackknife", crosspatch=True)
        counts = patched_counts_from_matrix(binning, patch_matrix_auto, auto=True)
        result = counts.sample_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_matrix_auto
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        # case: diagonal, autocorrelation
        config = ResamplingConfig(method="jackknife", crosspatch=False)
        counts = patched_counts_from_matrix(binning, patch_diag_auto, auto=True)
        result = counts.sample_sum(config)
        # compare to expected results, take only first bin
        data, jack, boot = expect_diag_auto
        assert result.data[0] == data
        npt.assert_equal(result.samples[:, 0], jack)

        subset = counts.patches[0]
        assert subset.sample_sum(config).samples.ndim == 2

    def test_bootstrap(self, binning, patch_matrix_full):
        counts = patched_counts_from_matrix(binning, patch_matrix_full, auto=False)
        with raises(NotImplementedError):
            counts.sample_sum(ResamplingConfig(method="bootstrap"))

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_bins(self, patched_counts_full, items):
        subset = patched_counts_full.bins[items]
        npt.assert_array_equal(
            subset.counts, apply_slice_ndim(patched_counts_full.counts, items, axis=2)
        )
        assert (subset.get_binning() == patched_counts_full.get_binning()[items]).all()

    def test_bins_iter(self, patched_counts_full):
        for i, _bin in enumerate(patched_counts_full.bins):
            npt.assert_array_equal(
                _bin.counts, apply_slice_ndim(patched_counts_full.counts, i, axis=2)
            )

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_patches(self, patched_counts_full, items):
        subset = patched_counts_full.patches[items]
        npt.assert_array_equal(
            subset.counts,
            apply_slice_ndim(patched_counts_full.counts, items, axis=(0, 1)),
        )
        assert (subset.get_binning() == patched_counts_full.get_binning()).all()

    def test_patches_iter(self, patched_counts_full):
        for i, _bin in enumerate(patched_counts_full.patches):
            npt.assert_array_equal(
                _bin.counts,
                apply_slice_ndim(patched_counts_full.counts, i, axis=(0, 1)),
            )

    def test_concatenate_patches(self, patched_counts_full, binning_next):
        merged = patched_counts_full.concatenate_patches(patched_counts_full)
        diag = np.diag(patched_counts_full.as_array()[:, :, 0])
        npt.assert_array_equal(
            np.diag(merged.as_array()[:, :, 0]), np.concatenate([diag, diag])
        )
        mod = modify_counts(patched_counts_full, binning=binning_next)
        with raises(ValueError):
            patched_counts_full.concatenate_patches(mod)

    def test_concatenate_bins(self, patched_counts_full, binning_next):
        appended = modify_counts(patched_counts_full, binning=binning_next)
        merged = patched_counts_full.concatenate_bins(appended)
        binned = patched_counts_full.as_array()[0, 0]
        npt.assert_array_equal(
            merged.as_array()[0, 0], np.concatenate([binned, binned])
        )
        with raises(ValueError):
            patched_counts_full.concatenate_bins(patched_counts_full)

    def test_add(self, patched_counts_full):
        summed = patched_counts_full + patched_counts_full
        npt.assert_array_equal(summed.counts, patched_counts_full.counts * 2)
        with raises(TypeError):
            patched_counts_full + 1
        with raises(ValueError):
            patched_counts_full + patched_counts_full.patches[:-1]
        with raises(ValueError):
            patched_counts_full + patched_counts_full.bins[:-1]

    def test_radd(self, patched_counts_full):
        assert (0 + patched_counts_full) == patched_counts_full
        assert sum([patched_counts_full]) == patched_counts_full
        with raises(TypeError):
            1 + patched_counts_full

    def test_mul(self, patched_counts_full):
        assert patched_counts_full * 2 == patched_counts_full + patched_counts_full
        with raises(TypeError):
            2 * patched_counts_full
        with raises(TypeError):
            patched_counts_full * True
        with raises(TypeError):
            patched_counts_full * [1, 2]

    def test_sum(self, patched_counts_full):
        assert patched_counts_full.sum() == np.sum(patched_counts_full.counts)


@fixture
def pair_count_result(patched_totals_full, patch_matrix_full):
    totals = patched_totals_full
    counts = patched_counts_from_matrix(
        totals.get_binning(), patch_matrix_full, auto=False
    )
    return paircounts.NormalisedCounts(total=totals, count=counts)


class TestNormalisedCounts:
    def test_init(self, patched_totals_full, patch_matrix_full):
        totals = patched_totals_full
        counts = patched_counts_from_matrix(
            totals.get_binning(), patch_matrix_full, auto=False
        )
        res = paircounts.NormalisedCounts(total=totals, count=counts)
        # wrong number of bins
        with raises(ValueError, match="bins"):
            counts = patched_counts_from_matrix(
                totals.get_binning()[:-1], patch_matrix_full, auto=False
            )
            paircounts.NormalisedCounts(total=totals, count=counts)
        # wrong number of patches
        with raises(ValueError, match="patches"):
            counts = patched_counts_from_matrix(
                totals.get_binning(), patch_matrix_full[:-1, :-1], auto=False
            )
            paircounts.NormalisedCounts(total=totals, count=counts)
        # just call once
        repr(res)

    def test_eq(self, pair_count_result):
        assert pair_count_result == pair_count_result
        assert pair_count_result != pair_count_result.bins[1:]
        assert pair_count_result != 1

    def test_sample(self, pair_count_result, expect_matrix_full):
        config = ResamplingConfig(method="jackknife")
        result = pair_count_result.sample(config)
        data, jack, boot = expect_matrix_full
        # since totals and counts are identical, expect ones everywhere
        npt.assert_equal(result.data[0], np.ones_like(data))
        npt.assert_equal(result.samples[:, 0], np.ones_like(jack))

    def test_hdf(self, pair_count_result, tmp_hdf5):
        pair_count_result.to_hdf(tmp_hdf5)
        assert "count" in tmp_hdf5
        assert "total" in tmp_hdf5
        restored = paircounts.NormalisedCounts.from_hdf(tmp_hdf5)
        assert restored.n_bins == pair_count_result.n_bins
        assert restored.n_patches == pair_count_result.n_patches
        pdt.assert_index_equal(restored.get_binning(), pair_count_result.get_binning())
        # compare total
        npt.assert_equal(restored.total.totals1, pair_count_result.total.totals1)
        npt.assert_equal(restored.total.totals2, pair_count_result.total.totals2)
        assert restored.total.auto == pair_count_result.total.auto
        # compare count
        for binA, binB in zip(
            restored.count._binning, pair_count_result.count._binning
        ):
            assert binA.left == binB.left
            assert binA.right == binB.right
        assert restored.total.auto == pair_count_result.total.auto

    def test_add(self, pair_count_result):
        summed = pair_count_result + pair_count_result
        assert pair_count_result.total == summed.total
        assert pair_count_result.count * 2 == summed.count
        with raises(TypeError):
            pair_count_result + 1
        # total does not agree
        new_total = paircounts.PatchedTotal(
            pair_count_result.get_binning(),
            pair_count_result.total.totals1 + 1,
            pair_count_result.total.totals2,
            auto=pair_count_result.auto,
        )
        new = paircounts.NormalisedCounts(pair_count_result.count, new_total)
        with raises(ValueError, match="total number"):
            pair_count_result + new

    def test_radd(self, pair_count_result):
        assert (0 + pair_count_result) == pair_count_result
        assert sum([pair_count_result]) == pair_count_result
        with raises(TypeError):
            1 + pair_count_result

    def test_mul(self, pair_count_result):
        mul = pair_count_result * 2
        assert mul.total == pair_count_result.total
        assert mul.count == pair_count_result.count * 2
        with raises(TypeError):
            pair_count_result * True

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_bins(self, pair_count_result, items):
        pc = pair_count_result
        assert pc.bins[items] == paircounts.NormalisedCounts(
            pc.count.bins[items], pc.total.bins[items]
        )

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_patches(self, pair_count_result, items):
        pc = pair_count_result
        assert pc.patches[items] == paircounts.NormalisedCounts(
            pc.count.patches[items], pc.total.patches[items]
        )

    def test_auto(self, pair_count_result):
        assert pair_count_result.auto == pair_count_result.count.auto

    def test_concatenate_patches(self, pair_count_result):
        merged = pair_count_result.concatenate_patches(pair_count_result)
        merged_count = pair_count_result.count.concatenate_patches(
            pair_count_result.count
        )
        merged_total = pair_count_result.total.concatenate_patches(
            pair_count_result.total
        )
        assert merged == paircounts.NormalisedCounts(
            count=merged_count, total=merged_total
        )
        with raises(TypeError, match="concat"):
            pair_count_result.concatenate_patches(pair_count_result.total)

    def test_concatenate_bins(self, pair_count_result, binning_next):
        shifted_count = modify_counts(pair_count_result.count, binning=binning_next)
        shifted_total = modify_totals(pair_count_result.total, binning=binning_next)
        shifted_pc = paircounts.NormalisedCounts(
            count=shifted_count, total=shifted_total
        )
        merged = pair_count_result.concatenate_bins(shifted_pc)
        merged_count = pair_count_result.count.concatenate_bins(shifted_count)
        merged_total = pair_count_result.total.concatenate_bins(shifted_total)
        assert merged == paircounts.NormalisedCounts(
            count=merged_count, total=merged_total
        )
        with raises(TypeError, match="concat"):
            pair_count_result.concatenate_bins(pair_count_result.total)


def test_pack_results(pair_count_result):
    pc = pair_count_result
    counts = dict(a=pc.count)
    assert paircounts.pack_results(counts, pc.total) == pc
    counts["b"] = pc.count
    result = paircounts.pack_results(counts, pc.total)
    for key in counts:
        assert result[key] == pc
