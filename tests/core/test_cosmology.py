import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15
from pytest import fixture

from yaw.core.cosmology import get_default_cosmology, r_kpc_to_angle


def test_default_cosmology():
    """this test is primarily a reminder that other tests with fail if the
    cosmolgy changes"""
    assert get_default_cosmology() == Planck15


def test_r_kpc_to_angle():
    require = np.array([0.00037711, 0.00377109])
    result = r_kpc_to_angle([500, 5000], 5.0, get_default_cosmology())
    npt.assert_almost_equal(result, require)
