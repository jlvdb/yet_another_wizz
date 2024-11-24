"""
Implements all code related to computing and expressing correlation functions
from a set of input data and random catalogs.

The main functions, autocorrelate and crosscorrelate, compute the correlation
functions stored as a set of pair counts, wrapped in a CorrFunc container.
The latter can be used to evaluate the pair counts with a correlation estimator
to obtain the actual measurement of the correlation function, expressed by a
CorrData object.
"""

from yaw.correlation.corrdata import CorrData
from yaw.correlation.corrfunc import CorrFunc
from yaw.correlation.measurements import autocorrelate, crosscorrelate
from yaw.correlation.paircounts import (
    NormalisedCounts,
    PatchedCounts,
    PatchedSumWeights,
)

__all__ = [
    "CorrData",
    "CorrFunc",
    "NormalisedCounts",
    "PatchedCounts",
    "PatchedSumWeights",
    "autocorrelate",
    "crosscorrelate",
]
