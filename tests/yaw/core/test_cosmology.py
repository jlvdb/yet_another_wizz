import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15
from pytest import mark

from yaw.core import cosmology


def test_default_cosmology():
    """this test is primarily a reminder that other tests with fail if the
    cosmolgy changes"""
    assert cosmology.get_default_cosmology() == Planck15


def test_r_kpc_to_angle():
    require = np.array([0.00037711, 0.00377109])
    result = cosmology.r_kpc_to_angle(
        [500, 5000], 5.0, cosmology.get_default_cosmology())
    npt.assert_almost_equal(result, require)


class TestBinFactory:

    @mark.skip
    def test_init(self):
        pass

    @mark.skip
    def test_linear(self):
        pass  # include .get()

    @mark.skip
    def test_comoving(self):
        pass  # include .get()

    @mark.skip
    def test_logspace(self):
        pass  # include .get()

    @mark.skip
    def test_check(self):
        pass
