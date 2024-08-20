"""This module implements the default values for the different configuration
objects. Each of the ``*Config`` objects has a corresponding class holding just
the default values, as listed below.
"""

__all__ = [
    "NotSet",
    "Scales",
    "Binning",
    "Configuration",
]


class _NotSet_meta(type):
    def __repr__(self) -> str:
        return "NotSet"  # pragma: no cover

    def __bool__(self) -> bool:
        return False


class NotSet(metaclass=_NotSet_meta):
    pass


# docs: render code below


class Scales:
    rweight = None
    rbin_num = 50


class Binning:
    method = "linear"
    zbin_num = 30


AutoBinning = Binning  # keep for backwards compatibilty


class Configuration:
    scales = Scales
    binning = Binning
    cosmology = "Planck15"
