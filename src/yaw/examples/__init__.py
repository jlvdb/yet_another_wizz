"""
This module provides precomputed example data products, which are loaded when
importing the module.

The data is based on spectroscopic data and randoms from the southern field the
2-degree Field Lensing Survey (2dFLenS, Blake et al. 2016, MNRAS, 462, 4240).

>>> from yaw import examples  # reads the data sets from disk
>>> examples.w_sp
CorrFunc(counts=dd|dr, auto=False, binning=11 bins @ (0.150...0.700], num_patches=11)
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
    "path_data",
    "path_rand",
]


_path = Path(__file__).parent

path_data = _path / "2dflens_kidss_data.pqt"
"""Path to a sample of 2dFLenS high-z data as Parquet file."""
path_rand = _path / "2dflens_kidss_rand_5x.pqt"
"""Path to a sample of 2dFLenS high-z randoms as Parquet file."""


w_sp = CorrFunc.from_file(_path / "cross.hdf")
"""Example data from a crosscorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""

w_ss = CorrFunc.from_file(_path / "auto_reference.hdf")
"""Example data from a reference sample autocorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""

w_pp = CorrFunc.from_file(_path / "auto_unknown.hdf")
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
