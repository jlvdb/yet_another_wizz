from astropy.cosmology import FlatwCDM, Planck15
from pytest import mark, raises

from yaw.config import utils
from yaw.core.cosmology import CustomCosmology, get_default_cosmology


class DummyCosmo(CustomCosmology):
    def comoving_distance(self, z):
        return 1.0

    def comoving_transverse_distance(self, z):
        return 1.0


def test_cosmology_to_yaml():
    with raises(TypeError):
        utils.cosmology_to_yaml(1)
    with raises(utils.ConfigError):
        utils.cosmology_to_yaml(DummyCosmo())
    with raises(utils.ConfigError):
        utils.cosmology_to_yaml(FlatwCDM(70, 0.3))
    assert utils.cosmology_to_yaml(Planck15) == Planck15.name


def test_yaml_to_cosmology():
    with raises(utils.ConfigError):
        utils.yaml_to_cosmology("my undefined cosmolgy")
    assert utils.yaml_to_cosmology(Planck15.name) == Planck15


def test_parse_cosmology():
    assert utils.parse_cosmology(None) == get_default_cosmology()
    assert utils.parse_cosmology(Planck15.name) == Planck15
    with raises(utils.ConfigError):
        utils.parse_cosmology("my undefined cosmolgy")
    with raises(utils.ConfigError):
        utils.parse_cosmology(123)
    assert utils.parse_cosmology(Planck15) == Planck15


@mark.parametrize("exception", [KeyError, TypeError])
def test_parse_section_error_propagates_unknown(exception):
    with raises(exception):
        try:
            raise exception
        except Exception as e:
            utils.parse_section_error(e, "some_section")


def test_parse_section_error_propagates_other_typeerror():
    with raises(TypeError):
        try:
            raise TypeError("some error")
        except Exception as e:
            utils.parse_section_error(e, "some_section")
