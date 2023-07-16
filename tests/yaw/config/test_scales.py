import numpy as np
from pytest import fixture, raises

from yaw.config import ScalesConfig
from yaw.config.utils import ConfigError
from yaw.core.cosmology import Scale


@fixture
def default_scales():
    return ScalesConfig(np.array([100, 200]), np.array([1000, 2000]))


class TestScalesConfig:
    def test_init_scalar(self):
        with raises(ConfigError):
            ScalesConfig(100, 100)

    def test_init_array(self, default_scales):
        with raises(ConfigError):
            ScalesConfig([100], [1000, 2000])
        with raises(ConfigError):
            ScalesConfig([100, 2000], [1000, 200])
        assert isinstance(default_scales.rmin, list)
        assert isinstance(default_scales.rmax, list)
        conf = ScalesConfig(np.array([100]), np.array([1000]))
        assert isinstance(conf.rmin, float)
        assert isinstance(conf.rmax, float)

    def test_init_mixed(self):
        with raises(ConfigError):
            ScalesConfig(100, [100])

    def test_iter(self, default_scales):
        scales = set(
            Scale(rmin, rmax)
            for rmin, rmax in zip(default_scales.rmin, default_scales.rmax)
        )
        assert scales == set(default_scales)

    def test_getitem(self, default_scales):
        assert default_scales[1] == Scale(
            default_scales.rmin[1], default_scales.rmax[1]
        )

    def test_eq(self, default_scales):
        scales = default_scales
        assert scales != ScalesConfig(scales.rmin[:1], scales.rmax[:1])
        assert scales != ScalesConfig(scales.rmin, scales.rmax, rweight=1.0)
        assert scales != ScalesConfig(scales.rmin, scales.rmax, rbin_num=11)
