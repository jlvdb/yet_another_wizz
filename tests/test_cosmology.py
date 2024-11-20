from astropy.cosmology import FLRW, Planck15
from pytest import raises

from yaw.cosmology import CustomCosmology, cosmology_is_equal, get_default_cosmology


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
