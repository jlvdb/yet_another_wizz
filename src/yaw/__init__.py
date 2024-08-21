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

from yaw._version import __version__, __version_tuple__
from yaw.catalog import Catalog
from yaw.config import Configuration
from yaw.corrfunc import CorrData, CorrFunc
from yaw.measurements import autocorrelate, crosscorrelate
from yaw.redshifts import HistData, RedshiftData

__all__ = [
    "__version__",
    "__version_tuple__",
    "Catalog",
    "Configuration",
    "CorrData",
    "CorrFunc",
    "HistData",
    "RedshiftData",
    "autocorrelate",
    "crosscorrelate",
]
