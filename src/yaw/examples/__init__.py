"""This module provides some precomputed example data, which are loaded when
importing the module, e.g.

>>> from yaw import examples  # reads the data sets from disk
>>> examples.w_sp
CorrFunc(n_bins=30, z='0.070...1.420', dd=True, dr=True, rd=False, rr=False, n_patches=64)
"""

from pathlib import Path

from yaw.corrfunc import CorrFunc

__all__ = [
    "w_sp",
    "w_ss",
    "w_pp",
    "normalised_counts",
    "patched_count",
    "patched_total",
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
(:obj:`~yaw.paircounts.NormalisedCounts` instance, from :obj:`w_sp.dd`)"""

patched_count = normalised_counts.counts
"""Example data for patch-wise pair counts
(:obj:`~yaw.paircounts.PatchedCount` instance, from :obj:`w_sp.dd.count`)"""

patched_total = normalised_counts.totals
"""Example data for patch-wise total number of objects
(:obj:`~yaw.paircounts.PatchedTotal` instance, from :obj:`w_sp.dd.total`)"""

del _path
