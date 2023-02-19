import numpy as np
import numpy.testing as npt
from pytest import fixture, raises


from yaw.core import paircounts


@fixture
def n_patches():
    return 5


@fixture
def n_bins():
    return 5


def add_bin_dimension(array, bin_no):
    return np.repeat(array, bin_no).reshape((*array.shape, bin_no))


@fixture
def total_data(n_patches, n_bins):
    return paircounts.PatchedTotal(
        add_bin_dimension(np.arange(0, n_patches), n_bins),
        add_bin_dimension(np.arange(1, n_patches + 1), n_bins),
        auto=False)


class TestPatchedTotal:

    def test_init(self, n_patches, n_bins, total_data):
        assert total_data.n_patches == n_patches
        assert total_data.n_bins == n_bins
        # now dimension mismatch
        with raises(ValueError):
            inst = paircounts.PatchedTotal(
                total_data.totals1, total_data.totals2[1:],
                auto=False)
        # missing dimesion
        with raises(ValueError):
            inst = paircounts.PatchedTotal(
                total_data.totals1[0], total_data.totals2[0],
                auto=False)

    def test_sum(self, total_data):
        pass
