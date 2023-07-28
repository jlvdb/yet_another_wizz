"""
yaw
===

*yet_another_wizz* is a python package to efficiently compute cross-correlation
redshifts, also know as clustering redshifts and is hosted on github:

- code: https://github.com/jlvdb/yet_another_wizz.git
- docs: https://yet-another-wizz.readthedocs.io/

The method allows to estimate the unknown redshift distribution of a galaxy
sample by correlating the on-sky positions with a reference sample with known
redshifts. This implementation is based on the single bin correlation
measurement of the correlation amplitude, introduced by Schmidt et al. (2013,
`arXiv:1303.0292 <https://arxiv.org/abs/1303.0292>`_).

Author: Jan Luca van den Busch
        (Ruhr-Universität Bochum, Astronomisches Institut)
"""


import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())  # noqa

from yaw.catalogs import NewCatalog
from yaw.config import Configuration, ResamplingConfig
from yaw.core.cosmology import Scale
from yaw.core.math import global_covariance
from yaw.correlation import CorrData, CorrFunc, autocorrelate, crosscorrelate
from yaw.randoms import UniformRandoms
from yaw.redshifts import HistData, RedshiftData

# isort: split
from yaw.deprecated.correlation.corrfuncs import CorrelationData, CorrelationFunction
from yaw.deprecated.correlation.paircounts import PairCountResult
from yaw.deprecated.redshifts import HistogramData

__all__ = [
    "NewCatalog",
    "Configuration",
    "ResamplingConfig",
    "Scale",
    "global_covariance",
    "CorrData",
    "CorrFunc",
    "autocorrelate",
    "crosscorrelate",
    "UniformRandoms",
    "HistData",
    "RedshiftData",
    # deprecated
    "CorrelationData",
    "CorrelationFunction",
    "HistogramData",
    "PairCountResult",
]
__version__ = "2.5.5"
