from yaw.config.default import NotSet


class TestNotSet:
    def test_is_false(self):
        assert bool(NotSet) is False
        if NotSet:
            assert 0
