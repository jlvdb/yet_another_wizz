import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15
from pytest import raises

from yaw.core import cosmology


def test_default_cosmology():
    """this test is primarily a reminder that other tests with fail if the
    cosmolgy changes"""
    assert cosmology.get_default_cosmology() == Planck15


def test_r_kpc_to_angle():
    require = np.array([0.00037711, 0.00377109])
    result = cosmology.r_kpc_to_angle(
        [500, 5000], 5.0, cosmology.get_default_cosmology()
    )
    npt.assert_almost_equal(result, require)


class TestScale:
    def test_init(self):
        with raises(ValueError):
            cosmology.Scale(10, 2)

    def test_mid(self):
        assert cosmology.Scale(10.0, 30.0).mid == 20.0

    def test_mid_log(self):
        assert cosmology.Scale(1.0, 100.0).mid_log == 10.0

    def test_to_radian(self):
        cosmo = cosmology.get_default_cosmology()
        z = 5.0
        npt.assert_array_equal(
            cosmology.Scale(500.0, 5000.0).to_radian(z, cosmo),
            cosmology.r_kpc_to_angle([500.0, 5000.0], z, cosmo),
        )


class TestBinFactory:
    def test_init(self):
        nbins = 30
        with raises(ValueError):
            cosmology.BinFactory(0.4, 0.1, nbins)
        assert (
            cosmology.BinFactory(0.1, 0.2, nbins).cosmology
            == cosmology.get_default_cosmology()
        )

    def test_linear(self):
        result = [1, 2, 3]
        factory = cosmology.BinFactory(1, 3, nbins=2)
        npt.assert_array_equal(factory.linear(), result)
        npt.assert_array_equal(factory.get("linear"), result)

    def test_comoving(self):
        result = np.array([0.1, 0.1972829, 0.3])  # from previous run
        factory = cosmology.BinFactory(0.1, 0.3, nbins=2)
        npt.assert_almost_equal(factory.comoving(), result)
        npt.assert_almost_equal(factory.get("comoving"), result)

    def test_logspace(self):
        result = np.exp([1, 2, 3]) - 1.0
        factory = cosmology.BinFactory(result[0], result[-1], nbins=2)
        # encountered machine precision issue on github actions
        npt.assert_array_almost_equal(factory.logspace(), result)
        npt.assert_array_almost_equal(factory.get("logspace"), result)

    def test_check(self):
        factory = cosmology.BinFactory(0.1, 0.2, nbins=9)
        with raises(ValueError):
            factory.get("my undefined method")
