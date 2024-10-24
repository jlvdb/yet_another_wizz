import numpy as np
from astropy.cosmology import FLRW, Planck15
from pytest import approx, raises

from yaw.utils.cosmology import (
    CustomCosmology,
    cosmology_is_equal,
    get_default_cosmology,
    separation_physical_to_angle,
)


class TestCosmology(CustomCosmology):
    def angular_diameter_distance(self, z):
        pass

    def comoving_distance(self, z):
        pass


def test_cosmology_is_equal():
    assert cosmology_is_equal(Planck15, Planck15)
    assert cosmology_is_equal(TestCosmology(), TestCosmology())
    assert not cosmology_is_equal(Planck15, TestCosmology())

    with raises(TypeError):
        cosmology_is_equal(Planck15, 1)
    with raises(TypeError):
        cosmology_is_equal(1, Planck15)


def test_get_default_cosmology():
    assert isinstance(get_default_cosmology(), FLRW)


def test_separation_physical_to_angle():
    redshifts = np.array([0.5, 1.0])
    arcmin_per_kpc = 1.0 / Planck15.kpc_proper_per_arcmin(redshifts)
    rad_per_kpc = np.deg2rad(arcmin_per_kpc.value / 60.0)

    for z, expect in zip(redshifts, rad_per_kpc):
        result = separation_physical_to_angle(1.0, z, cosmology=Planck15)
        assert approx(expect) == result


def test_separation_physical_to_angle_shape():
    scalar = 1.0
    array = np.array([1.0, 2.0])

    result = separation_physical_to_angle(scalar, array, cosmology=Planck15)
    assert result.shape == (2, 1)
    result = separation_physical_to_angle(array, scalar, cosmology=Planck15)
    assert result.shape == (1, 2)
    result = separation_physical_to_angle(array, array, cosmology=Planck15)
    assert result.shape == (2, 2)
