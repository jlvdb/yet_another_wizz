"""This module implements the computation of correlation functions with
:func:`crosscorrelate` and :func:`autocorrelate` from input data catalogues, see
:mod:`yaw.catalogs`.

These measurement methods return the main class, the correlation function
:obj:`CorrFunc` container. It holds the normalised pair counts (in
:obj:`~paircounts.NormalisedCounts` containers) computed within each spatial
patch and bins of redshift. The actual correlation function values and its
uncertainty (from resampling the spatial patches) can be computed using the
:meth:`CorrFunc.sample()`, which returns a :obj:`CorrData` container.

For the conversion of the correlation functions to a redshift estimate refer to
the :mod:`yaw.redshifts` module.
"""

from yaw.correlation.corrfuncs import (
    CorrData,
    CorrFunc,
    add_corrfuncs,
    autocorrelate,
    crosscorrelate,
)

__all__ = ["CorrData", "CorrFunc", "add_corrfuncs", "autocorrelate", "crosscorrelate"]
