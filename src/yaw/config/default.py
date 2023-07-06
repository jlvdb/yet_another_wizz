"""This module implements the default values for the different configuration
objects. Each of the ``*Config`` objects has a corresponding class holding just
the default values, as listed below.
"""

class _NotSet_meta(type):

    def __repr__(self) -> str:
        return "NotSet"  # pragma: no cover

    def __bool__(self) -> bool:
        return False


class NotSet(metaclass=_NotSet_meta):
    pass


class Scales:
    rweight = None
    rbin_num = 50


class AutoBinning:
    method = "linear"
    zbin_num = 30


class Backend:
    thread_num = None
    crosspatch = True
    rbin_slop = 0.01


class Configuration:
    scales = Scales
    binning = AutoBinning
    backend = Backend
    cosmology = "Planck15"


class Resampling:
    method = "jackknife"
    crosspatch = True
    n_boot = 500
    global_norm = False
    seed = 12345


backend = "scipy"
