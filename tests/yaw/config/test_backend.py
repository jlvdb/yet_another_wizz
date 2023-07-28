import os

from pytest import raises

from yaw.config import BackendConfig


class TestBackendConfig:
    def test_default_threads(self):
        assert BackendConfig().thread_num == os.cpu_count()

    def test_modify(self):
        substitutes = dict(
            thread_num=20,
            crosspatch=False,
            rbin_slop=-1.0,
        )
        for param, value in substitutes.items():
            conf = BackendConfig().modify(**{param: value})
            assert getattr(conf, param) == value

    def test_get_threads(self):
        conf = BackendConfig(thread_num=10)
        assert conf.get_threads() == 10
        assert conf.get_threads(max=2) == 2
        with raises(ValueError):
            conf.get_threads(0)
