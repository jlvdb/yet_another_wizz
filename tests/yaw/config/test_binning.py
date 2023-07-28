import numpy as np
import numpy.testing as npt
from pytest import fixture, raises

from yaw.config.binning import BinningConfig
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
    return BinningConfig.create(zbins=bin_edges)


@fixture
def auto_binning(bin_props):
    return BinningConfig.create(**bin_props)


class TestBinningConfig:
    def test_init(self, bin_edges, manual_binning):
        npt.assert_array_equal(bin_edges, manual_binning.zbins)
        assert manual_binning.method == "manual"
        with raises(ValueError):
            BinningConfig(zbins=bin_edges[::-1])
        with raises(ConfigError):
            BinningConfig(zbins=[0.1])

    def test_pass_repr(self, manual_binning):
        str(manual_binning)

    def test_eq(self, bin_edges, bin_props, auto_binning, manual_binning):
        new_bins = bin_edges.copy()
        new_bins[-1] = 2.0
        assert BinningConfig.create(zbins=bin_edges) == manual_binning
        assert not BinningConfig.create(zbins=new_bins) == manual_binning

        assert BinningConfig.create(**bin_props) == auto_binning
        new_props = dict(
            zmin=bin_props["zmin"] + 0.001,
            zmax=bin_props["zmax"] + 0.001,
            zbin_num=bin_props["zbin_num"] + 1,
            method="comoving",
        )
        for param, value in new_props.items():
            the_props = {k: v for k, v in bin_props.items()}
            the_props[param] = value
            assert not BinningConfig.create(**the_props) == auto_binning

        assert auto_binning != 1

    def test_properties(self, bin_props, auto_binning, manual_binning):
        assert manual_binning.zmin == bin_props["zmin"]
        assert manual_binning.zmax == bin_props["zmax"]
        assert manual_binning.zbin_num == bin_props["zbin_num"]
        assert manual_binning.method == "manual"
        assert manual_binning.is_manual

        assert auto_binning.zmin == bin_props["zmin"]
        assert auto_binning.zmax == bin_props["zmax"]
        assert auto_binning.zbin_num == bin_props["zbin_num"]
        assert auto_binning.method == "linear"
        assert not auto_binning.is_manual

    def test_create_auto(self, bin_props, auto_binning):
        assert auto_binning == BinningConfig.create(**bin_props)
        the_props = {k: v for k, v in bin_props.items()}
        the_props["method"] = None
        with raises(ValueError):
            BinningConfig.create(**the_props)

    def test_create_manual(self, bin_edges, manual_binning):
        assert manual_binning == BinningConfig.create(zbins=bin_edges)

    def test_create_all_arg(self, bin_edges, bin_props, manual_binning):
        assert isinstance(
            BinningConfig.create(zbins=bin_edges, **bin_props), BinningConfig
        )

    def test_create_no_z_arg(self, bin_edges, bin_props, auto_binning):
        with raises(ConfigError):
            BinningConfig.create(zmin=0.1)
        with raises(ConfigError):
            BinningConfig.create(zmax=0.1)
        with raises(ConfigError):
            BinningConfig.create()

    def test_modify_auto(self, bin_props, auto_binning, manual_binning):
        assert auto_binning == auto_binning.modify(**bin_props)
        assert auto_binning == manual_binning.modify(**bin_props)

        for param, value in bin_props.items():
            conf = auto_binning.modify(**{param: value})
            assert getattr(conf, param) == value

        for param in ("zmin", "zmax"):
            with raises(ConfigError):
                manual_binning.modify(**{param: bin_props[param]})

    def test_modify_manual(self, bin_edges, auto_binning, manual_binning):
        assert manual_binning == auto_binning.modify(zbins=bin_edges)
        assert manual_binning == manual_binning.modify(zbins=bin_edges)

    def test_modify_all_arg(self, bin_edges, bin_props, auto_binning, manual_binning):
        assert auto_binning.modify(zbins=bin_edges, **bin_props).is_manual
        assert manual_binning.modify(zbins=bin_edges, **bin_props).is_manual

    def test_modify_no_z_arg(self, manual_binning):
        with raises(ConfigError):
            manual_binning.modify(zmin=0.1)
        with raises(ConfigError):
            manual_binning.modify(zmax=0.1)

    def test_to_dict(self, bin_edges, auto_binning, manual_binning):
        the_dict = manual_binning.to_dict()
        assert set(the_dict.keys()) == {"zbins", "method"}
        npt.assert_array_equal(the_dict["zbins"], bin_edges)

        the_dict = auto_binning.to_dict()
        assert set(the_dict.keys()) == {"zmin", "zmax", "zbin_num", "method"}

    def test_from_dict(self, bin_edges, bin_props, auto_binning, manual_binning):
        the_dict = dict(zbins=bin_edges)
        assert BinningConfig.from_dict(the_dict) == manual_binning

        the_dict = {key: val for key, val in bin_props.items()}
        assert BinningConfig.from_dict(the_dict) == auto_binning
