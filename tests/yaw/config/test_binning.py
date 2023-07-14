import numpy as np
import numpy.testing as npt
from pytest import fixture, raises

from yaw.config.binning import (
    AutoBinningConfig,
    ManualBinningConfig,
    make_binning_config,
)
from yaw.config.utils import ConfigError


@fixture
def bin_edges():
    return np.array([0.1, 0.2, 0.3])


@fixture
def bin_props(bin_edges):
    props = dict(
        zmin=bin_edges.min(),
        zmax=bin_edges.max(),
        zbin_num=len(bin_edges) - 1,
        method="linear",
    )
    return props


@fixture
def manual_binning(bin_edges):
    return ManualBinningConfig(bin_edges)


@fixture
def auto_binning(bin_props):
    return AutoBinningConfig.generate(**bin_props)


class TestManualBinningConfig:
    def test_init(self, bin_edges, manual_binning):
        npt.assert_array_equal(bin_edges, manual_binning.zbins)
        assert manual_binning.method == "manual"
        with raises(ValueError):
            ManualBinningConfig(bin_edges[::-1])
        with raises(ConfigError):
            ManualBinningConfig([0.1])

    def test_eq(self, bin_edges, manual_binning):
        new_bins = bin_edges.copy()
        new_bins[-1] = 2.0
        assert ManualBinningConfig(bin_edges) == manual_binning
        assert not ManualBinningConfig(new_bins) == manual_binning

    def test_properties(self, bin_props, manual_binning):
        assert manual_binning.zmin == bin_props["zmin"]
        assert manual_binning.zmax == bin_props["zmax"]
        assert manual_binning.zbin_num == bin_props["zbin_num"]

    def test_to_dict(self, bin_edges, manual_binning):
        the_dict = manual_binning.to_dict()
        assert set(the_dict.keys()) == {"zbins", "method"}
        npt.assert_array_equal(the_dict["zbins"], bin_edges)

    def test_from_dict(self, bin_edges, manual_binning):
        the_dict = dict(zbins=bin_edges)
        assert ManualBinningConfig.from_dict(the_dict) == manual_binning
        # provide arugments only consumed by AutoBinningConfig
        the_dict.update(dict(zmin=bin_edges.min(), zmax=bin_edges.max()))
        with raises(TypeError):
            ManualBinningConfig.from_dict(the_dict)


class TestAutoBinningConfig:
    def test_generate(self, bin_edges, auto_binning):
        method = "linear"
        binning = AutoBinningConfig(bin_edges, method=method)
        npt.assert_array_equal(bin_edges, binning.zbins)
        assert binning.method == method
        npt.assert_array_equal(bin_edges, auto_binning.zbins)

    def test_eq(self, bin_props, auto_binning):
        assert AutoBinningConfig.generate(**bin_props) == auto_binning
        new_props = dict(
            zmin=bin_props["zmin"] + 0.001,
            zmax=bin_props["zmax"] + 0.001,
            zbin_num=bin_props["zbin_num"] + 1,
            method="comoving",
        )
        for param, value in new_props.items():
            the_props = {k: v for k, v in bin_props.items()}
            the_props[param] = value
            assert not AutoBinningConfig.generate(**the_props) == auto_binning

    def test_properties(self, bin_props, auto_binning):
        assert auto_binning.zmin == bin_props["zmin"]
        assert auto_binning.zmax == bin_props["zmax"]
        assert auto_binning.zbin_num == bin_props["zbin_num"]

    def test_to_dict(self, auto_binning):
        the_dict = auto_binning.to_dict()
        assert set(the_dict.keys()) == {"zmin", "zmax", "zbin_num", "method"}

    def test_from_dict(self, bin_props, auto_binning):
        the_dict = {key: val for key, val in bin_props.items()}
        assert AutoBinningConfig.from_dict(the_dict) == auto_binning
        # provide arugments only consumed by ManualBinningConfig
        the_dict.update(dict(zbins=bin_edges))
        with raises(TypeError):
            AutoBinningConfig.from_dict(the_dict)


def test_make_binning_config_auto(bin_props, auto_binning):
    assert auto_binning == make_binning_config(**bin_props)
    the_props = {k: v for k, v in bin_props.items()}
    the_props["method"] = None
    with raises(ValueError):
        make_binning_config(**the_props)


def test_make_binning_config_manual(bin_edges, manual_binning):
    assert manual_binning == make_binning_config(zbins=bin_edges)


def test_make_binning_config_all_arg(bin_edges, bin_props, manual_binning):
    assert isinstance(
        make_binning_config(zbins=bin_edges, **bin_props), AutoBinningConfig
    )


def test_make_binning_config_no_z_arg(bin_edges, bin_props, auto_binning):
    with raises(ConfigError):
        make_binning_config(zmin=0.1)
    with raises(ConfigError):
        make_binning_config(zmax=0.1)
    with raises(ConfigError):
        make_binning_config()
