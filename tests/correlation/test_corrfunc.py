import numpy as np
import numpy.testing as npt
import pandas as pd
from pytest import fixture, raises

from yaw.correlation import corrfunc


@fixture
def binning():
    return pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0, 1.5])


@fixture
def test_data_samples(binning):
    """
    Construct some fake data with value 4 and samples that yield a variance of
    4 2/3 for bootstrapping and 24 for jackknifing. The covariance matrix is
    constant.
    """
    n_bins, n_samples = len(binning), 7
    data = np.full(n_bins, 4)
    samples = np.repeat(np.arange(1, 8), n_bins).reshape((n_samples, n_bins))
    return binning, data, samples


@fixture
def stats_jackknife(binning):
    var = 24.
    err = np.full(len(binning), np.sqrt(var))
    cov = np.full((len(binning), len(binning)), var)
    cor = np.ones_like(cov)
    return err, cov, cor


@fixture
def stats_bootstrap(binning):
    var = 4.+2./3.
    err = np.full(len(binning), np.sqrt(var))
    cov = np.full((len(binning), len(binning)), var)
    cor = np.ones_like(cov)
    return err, cov, cor


@fixture
def correlation_data(test_data_samples):
    binning, data, samples = test_data_samples
    return corrfunc.CorrelationData(
        binning, data, samples, method="jackknife")


class TestCorrelationData:

    def test_stats(
            self, test_data_samples, stats_jackknife, stats_bootstrap):
        binning, data, samples = test_data_samples
        # jackknife
        err, cov, cor = stats_jackknife
        cd = corrfunc.CorrelationData(
            binning, data, samples, method="jackknife")
        npt.assert_allclose(cd.error, err)
        npt.assert_allclose(cd.covariance, cov)
        npt.assert_allclose(cd.get_correlation().to_numpy(), cor)
        # bootstrap
        err, cov, cor = stats_bootstrap
        cd = corrfunc.CorrelationData(
            binning, data, samples, method="bootstrap")
        npt.assert_allclose(cd.error, err)
        npt.assert_allclose(cd.covariance, cov)
        npt.assert_allclose(cd.get_correlation().to_numpy(), cor)

    def test_io(self, tmp_path_factory, correlation_data):
        path = tmp_path_factory.mktemp("corr_data_file") / "testfiles"
        correlation_data.to_files(path)
        restored = correlation_data.from_files(path)
        npt.assert_allclose(correlation_data.data, restored.data)
        npt.assert_allclose(correlation_data.samples, restored.samples)
