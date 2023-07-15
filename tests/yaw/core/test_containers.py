import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pytest import fixture, raises

from yaw.core import containers


@fixture
def binning():
    return pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0, 1.5])


@fixture
def test_data_samples(binning):
    n_bins, n_samples = len(binning), 7
    data = np.full(n_bins, 4)
    samples = np.repeat(np.arange(1, 8), n_bins).reshape((n_samples, n_bins))
    return binning, data, samples


class TestSampledData:
    def test_init(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = containers.SampledData(binning, data, samples, method="jackknife")
        # binning and data mismatch
        with raises(ValueError, match="unexpected *"):
            containers.SampledData(binning, data[1:], samples, method="jackknife")
        # wrong sample shape
        with raises(ValueError, match="number of bins*"):
            samples1 = samples[:, :1]
            containers.SampledData(binning, data, samples1, method="jackknife")
        # check some of the representation
        assert f"n_bins={samples.shape[1]}" in repr(sd)
        assert f"n_samples={samples.shape[0]}" in repr(sd)  # tests .n_samples
        # wrong sampling method
        with raises(ValueError):
            containers.SampledData(binning, data, samples, method="garbage")

    def test_getters(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = containers.SampledData(binning, data, samples, method="jackknife")
        # test shape and index
        pdt.assert_index_equal(binning, sd.get_data().index)
        pdt.assert_index_equal(binning, sd.get_samples().index)
        assert sd.get_data().shape == data.shape
        assert sd.get_samples().shape == samples.T.shape

    def test_compatible(self, test_data_samples):
        binning, data, samples = test_data_samples
        sd = containers.SampledData(binning, data, samples, method="jackknife")
        # trivial case
        assert sd.is_compatible(sd)
        # wrong type
        with raises(TypeError):
            sd.is_compatible(1)
        # different binning
        binning2 = pd.IntervalIndex.from_breaks(np.linspace(1.0, 2.0, len(binning) + 1))
        sd2 = containers.SampledData(binning2, data, samples, method="jackknife")
        assert not sd.is_compatible(sd2)
        # different number of samples
        sd2 = containers.SampledData(binning, data, samples[:-1], method="jackknife")
        assert not sd.is_compatible(sd2)
        # different method
        sd2 = containers.SampledData(binning, data, samples, method="bootstrap")
        assert not sd.is_compatible(sd2)

    def test_binning(self, test_data_samples):
        # methods defined in BinnedQuantity
        binning, data, samples = test_data_samples
        sd = containers.SampledData(binning, data, samples, method="jackknife")
        assert sd.n_bins == len(data)
        npt.assert_equal(sd.mids, np.array([0.25, 0.75, 1.25]))
        npt.assert_equal(sd.edges, np.array([0.0, 0.5, 1.0, 1.5]))
        npt.assert_equal(sd.dz, np.array([0.5, 0.5, 0.5]))
