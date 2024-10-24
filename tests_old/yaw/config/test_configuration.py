import numpy as np
import numpy.testing as npt
from astropy.cosmology import WMAP9, Planck15
from pytest import fixture, mark, raises

from yaw.config import BackendConfig, BinningConfig, Configuration
from yaw.config.utils import ConfigError
from yaw.core.cosmology import get_default_cosmology


@fixture
def default_kwargs():
    return dict(rmin=10.0, rmax=100.0, zmin=0.1, zmax=0.3)


@fixture
def default_config(default_kwargs):
    return Configuration.create(**default_kwargs)


@fixture
def default_dict(default_config):
    return default_config.to_dict()


@fixture
def flat_dict(default_dict):
    flat_dict = dict(cosmology=default_dict["cosmology"])
    flat_dict.update(default_dict["scales"])
    flat_dict.update(default_dict["binning"])
    flat_dict.update(default_dict["backend"])
    return flat_dict


@fixture
def manual_config(default_kwargs):
    kwargs = dict(
        rmin=default_kwargs["rmin"],
        rmax=default_kwargs["rmax"],
        zbins=np.array([0.1, 0.3, 0.5]),
    )
    return Configuration.create(**kwargs)


class TestConfigurationCreate:
    def test_default(self, default_config, default_kwargs):
        assert default_config.cosmology is get_default_cosmology()
        assert default_config.scales.rmin == default_kwargs["rmin"]
        assert default_config.scales.rmax == default_kwargs["rmax"]
        assert default_config.binning.zmin == default_kwargs["zmin"]
        assert default_config.binning.zmax == default_kwargs["zmax"]

    def test_no_rmin(self):
        kwargs = dict(rmax=100.0, zmin=0.1, zmax=0.3)
        with raises(TypeError):
            Configuration.create(**kwargs)

    def test_no_zmin(self):
        kwargs = dict(rmin=10.0, rmax=100.0, zmax=0.3)
        with raises(ConfigError):
            Configuration.create(**kwargs)

    def test_cosmo_parsing(self):
        kwargs = dict(cosmology="Planck15", rmin=10.0, rmax=100.0, zmin=0.1, zmax=0.3)
        conf = Configuration.create(**kwargs)
        assert conf.cosmology is Planck15
        assert conf.cosmology is not WMAP9

    def test_zbins(self):
        bins = np.array([0.1, 0.3, 0.5])
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zbins=bins,
        )
        conf = Configuration.create(**kwargs)
        npt.assert_array_equal(conf.binning.zbins, bins)

    def test_zbins_zmin_none(self):
        bins = np.array([0.1, 0.3, 0.5])
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zbins=bins,
            zmin=None,
        )
        conf = Configuration.create(**kwargs)
        npt.assert_array_equal(conf.binning.zbins, bins)

    def test_zbins_method_none(self):
        bins = np.array([0.1, 0.3, 0.5])
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zbins=bins,
            method=None,
        )
        conf = Configuration.create(**kwargs)
        npt.assert_array_equal(conf.binning.zbins, bins)

    def test_autoz_zbins_none(self):
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zmin=0.1,
            zmax=0.3,
            zbins=None,
        )
        assert Configuration.create(**kwargs).binning.zmin == 0.1

    def test_autoz_linear(self):
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zmin=0.1,
            zmax=0.3,
            zbin_num=2,
            method="linear",
        )
        conf = Configuration.create(**kwargs)
        assert conf.binning.method == "linear"
        assert conf.binning.zbins[1] == 0.2

    def test_autoz_comoving(self):
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zmin=0.1,
            zmax=0.3,
            zbin_num=2,
            method="comoving",
        )
        conf = Configuration.create(**kwargs)
        assert conf.binning.method == "comoving"
        assert conf.binning.zbins[1] != 0.2

    def test_zbin_num(self):
        kwargs = dict(
            rmin=10.0,
            rmax=100.0,
            zmin=0.1,
            zmax=0.3,
            zbin_num=10,
        )
        assert len(Configuration.create(**kwargs).binning.zbins) == 11


class TestConfigurationModify:
    def test_copy(self, default_config):
        assert default_config.modify() == default_config

    def test_parse_cosmology(self, default_config):
        assert default_config.modify(cosmology="WMAP9").cosmology == WMAP9
        assert default_config.modify(cosmology=WMAP9).cosmology == WMAP9

    def test_update_scales(self, default_config):
        substitutes = dict(
            rmin=20.0,
            rmax=200.0,
            rweight=1.0,
            rbin_num=10,
        )
        for param, value in substitutes.items():
            conf = default_config.modify(**{param: value})
            assert getattr(conf.scales, param) == value

    def test_update_binning_auto_update_zbins(self, default_config, manual_config):
        zbins = manual_config.binning.zbins
        assert default_config.modify(zbins=zbins).binning == manual_config.binning

    def test_update_binning_auto_update_other(self, default_config):
        auto_config: Configuration = default_config.modify()
        auto_substitutes = dict(
            zmin=0.001,
            zmax=10.0,
            zbin_num=5,
            method="comoving",
        )
        for param, value in auto_substitutes.items():
            conf = auto_config.modify(**{param: value})
            auto_kwargs = auto_config.to_dict()["binning"]
            auto_kwargs[param] = value
            assert conf.binning == BinningConfig.create(**auto_kwargs)

    def test_update_binning_manual_update_zbins(self, manual_config):
        zbins = np.linspace(1, 10)
        assert manual_config.modify(zbins=zbins).binning == BinningConfig.create(
            zbins=zbins
        )

    def test_update_binning_manual_all_other(self, manual_config, default_dict):
        bin_dict = default_dict["binning"]
        kwargs = dict(zmin=bin_dict["zmin"], zmax=bin_dict["zmax"])
        # only z limits
        expect = BinningConfig.create(**bin_dict)
        got = manual_config.modify(**kwargs)
        assert expect == got.binning
        # add method
        added_dict = {k: v for k, v in bin_dict.items() if k != "method"}
        expect = BinningConfig.create(**added_dict, method="comoving")
        got = manual_config.modify(**kwargs, method="comoving")
        assert expect == got.binning
        # add zbin_num
        added_dict = {k: v for k, v in bin_dict.items() if k != "zbin_num"}
        expect = BinningConfig.create(**added_dict, zbin_num=5)
        got = manual_config.modify(**kwargs, zbin_num=5)
        assert expect == got.binning

    @mark.parametrize("param", ["zmin", "zmax"])
    def test_update_binning_manual_some_other(self, manual_config, param):
        with raises(ConfigError):
            manual_config.modify(**{param: 0.1})

    def test_update_backend(self, default_config):
        substitutes = dict(
            thread_num=100,
            crosspatch=False,
            rbin_slop=10.0,
        )
        for param, value in substitutes.items():
            conf = default_config.modify(**{param: value})
            assert getattr(conf.backend, param) == value

    def test_update_cosmology(self, default_config):
        assert default_config.modify(cosmology=WMAP9).cosmology == WMAP9


class TestConfiguration:
    def test_parse_cosmology(self, default_config):
        conf = Configuration(
            scales=default_config.scales,
            binning=default_config.scales,
            backend=default_config.backend,
            cosmology=None,
        )
        assert conf.cosmology == get_default_cosmology()
        conf = Configuration(
            scales=default_config.scales,
            binning=default_config.scales,
            backend=default_config.backend,
        )
        assert conf.cosmology == get_default_cosmology()
        with raises(ConfigError):
            Configuration(
                scales=default_config.scales,
                binning=default_config.scales,
                backend=default_config.backend,
                cosmology=1,
            )

    def test_from_dict(self, default_dict, flat_dict):
        # make sure that .create and .from_dict produce the same results
        created = Configuration.create(**flat_dict)
        restored = Configuration.from_dict(default_dict)
        assert created == restored

    def test_from_dict_missing_sections(self, default_dict):
        sca = default_dict["scales"]
        bng = default_dict["binning"]
        bck = default_dict["backend"]
        # drop cosmology
        conf = Configuration.from_dict(dict(scales=sca, binning=bng, backend=bck))
        assert conf.cosmology == get_default_cosmology()
        # drop scales
        with raises(ConfigError, match="scales"):
            Configuration.from_dict(dict(binning=bng, backend=bck))
        # drop binning
        with raises(ConfigError, match="binning"):
            Configuration.from_dict(dict(scales=sca, backend=bck))
        # drop backend
        conf = Configuration.from_dict(dict(scales=sca, binning=bng))
        assert conf.backend == BackendConfig()

    @mark.parametrize("section", ["scales", "binning", "backend"])
    def test_from_dict_extra_item(self, default_dict, section):
        extra_dict = {k: v for k, v in default_dict.items()}
        extra_dict[section]["extra_key"] = None
        with raises(ConfigError, match="unknown option"):
            Configuration.from_dict(extra_dict)

    @mark.parametrize(
        "section,item",
        [
            ("scales", "rmin"),
            ("scales", "rmax"),
        ],
    )
    def test_from_dict_missing_item(self, default_dict, section, item):
        miss_dict = {k: v for k, v in default_dict.items()}
        miss_dict[section].pop(item)
        with raises(ConfigError, match="missing option"):
            Configuration.from_dict(miss_dict)

    def test_from_dict_extra_section(self, default_dict):
        the_dict = {k: v for k, v in default_dict.items()}
        the_dict["extra_section"] = dict()
        with raises(ConfigError, match="unknown section"):
            Configuration.from_dict(the_dict)

    def test_yaml(self, default_config, tmp_path):
        # implicitly tests that .to_dict -> .from_dict is consistent
        f = tmp_path / "config_dump.yaml"
        default_config.to_yaml(f)
        assert default_config == Configuration.from_yaml(f)
