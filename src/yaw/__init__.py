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

from yaw.catalog import Catalog
from yaw.config import Configuration, ResamplingConfig
from yaw.containers import CorrData, HistData, RedshiftData
from yaw.corrfunc import CorrFunc
from yaw.measurements import autocorrelate, crosscorrelate

__all__ = [
    "Catalog",
    "Configuration",
    "CorrData",
    "CorrFunc",
    "HistData",
    "ResamplingConfig",
    "RedshiftData",
    "autocorrelate",
    "crosscorrelate",
]
__version__ = "3.0.0"
