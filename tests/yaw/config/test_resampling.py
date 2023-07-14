import numpy as np
import numpy.testing as npt
from pytest import mark, raises

from yaw.config import OPTIONS, ResamplingConfig
from yaw.config.utils import ConfigError


class TestResamplingConfig:
    def test_init(self):
        with raises(ConfigError):
            ResamplingConfig(method="my undefined method")

    @mark.parametrize("method", OPTIONS.method)
    def test_n_patches(self, method):
        conf = ResamplingConfig(method=method)
        assert conf.n_patches is None
        n_patches = 10
        conf.get_samples(n_patches)
        assert conf.n_patches == n_patches

    def test__generate_jackknife(self):
        samples = ResamplingConfig()._generate_jackknife(n_patches=3)
        npt.assert_array_equal(samples.sum(axis=1), [3, 2, 1])

    def test__generate_bootstrap(self):
        conf = ResamplingConfig(method="bootstrap", seed=12345, n_boot=10)
        samples = conf._generate_bootstrap(n_patches=3)
        expect = np.array([4, 2, 5, 3, 2, 2, 5, 4, 4, 1])
        npt.assert_array_equal(samples.sum(axis=1), expect)

    def test_get_samples(self):
        # tested sampling methods above
        conf = ResamplingConfig()
        conf.get_samples(n_patches=3)
        with raises(ValueError):
            conf.get_samples(4)

    def test_reset(self):
        conf = ResamplingConfig()
        conf.get_samples(n_patches=3)
        conf.reset()
        assert conf._resampling_idx is None

        kwargs = dict(method="jackknife", crosspatch=False, global_norm=True)
        conf = ResamplingConfig(**kwargs)
        assert ResamplingConfig.from_dict(conf.to_dict()) == conf

        kwargs = dict(
            method="bootstrap", crosspatch=False, n_boot=10, global_norm=True, seed=321
        )
        conf = ResamplingConfig(**kwargs)
        assert ResamplingConfig.from_dict(conf.to_dict()) == conf
