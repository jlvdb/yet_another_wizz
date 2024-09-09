from pytest import mark

from yaw.config import OPTIONS


class TestOptions:
    @mark.parametrize(
        "parameter", ["backend", "binning", "cosmology", "kind", "merge", "method"]
    )
    def test_properties_are_tuple(self, parameter):
        option_values = getattr(OPTIONS, parameter)
        assert isinstance(option_values, tuple)
