"""
This module provides some precomputed example data, which are loaded when
importing the module.

>>> from yaw import examples  # reads the data sets from disk
>>> examples.w_sp
CorrFunc(counts=dd|dr, auto=False, num_bins=30, num_patches=64)
"""

from pathlib import Path

from yaw.correlation.corrfunc import CorrFunc

__all__ = [
    "w_sp",
    "w_ss",
    "w_pp",
    "normalised_counts",
    "patched_count",
    "patched_sum_weights",
]


_path = Path(__file__).parent

w_sp = CorrFunc.from_file(_path / "cross_1.hdf")
"""Example data from a crosscorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""

w_ss = CorrFunc.from_file(_path / "auto_reference.hdf")
"""Example data from a reference sample autocorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""

w_pp = CorrFunc.from_file(_path / "auto_unknown_1.hdf")
"""Example data from a unknown sample autocorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""


normalised_counts = w_sp.dd
"""Example data for patch-wise, normalised pair counts
(:obj:`~yaw.correlation.paircounts.NormalisedCounts` instance, from :obj:`w_sp.dd`)"""

patched_count = normalised_counts.counts
"""Example data for patch-wise pair counts
(:obj:`~yaw.correlation.paircounts.PatchedCount` instance, from :obj:`w_sp.dd.count`)"""

patched_sum_weights = normalised_counts.sum_weights
"""Example data for patch-wise sum of object weights
(:obj:`~yaw.correlation.paircounts.PatchedSumWeights` instance, from :obj:`w_sp.dd.sum_weights`)"""
