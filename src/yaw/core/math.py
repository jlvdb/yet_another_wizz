"""This module implements some math and array related functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

from ._math import _rebin

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

    from yaw.core.containers import SampledData

__all__ = [
    "array_equal",
    "outer_triu_sum",
    "apply_bool_mask_ndim",
    "apply_slice_ndim",
    "sgn",
    "cov_from_samples",
    "global_covariance",
    "corr_from_cov",
    "rebin",
    "shift_histogram",
]


_Tarr = TypeVar("_Tarr", bound=NDArray)


def array_equal(arr1: NDArray, arr2: NDArray) -> bool:
    """Check if the shape and array elements of two numpy array are equal."""
    return (
        isinstance(arr1, np.ndarray)
        and isinstance(arr2, np.ndarray)
        and arr1.shape == arr2.shape
        and (arr1 == arr2).all()
    )


def outer_triu_sum(
    a: ArrayLike, b: ArrayLike, *, k: int = 0, axis: int | None = None
) -> NDArray:
    """Compute the sum over the upper triangle of the outer product.

    Shapes of input array must be identical. Equivalent to

    >>> np.triu(np.outer(a, b), k).sum(axis)

    but supports extra dimensions in a and b and does not construct the full
    outer product matrix in memory.

    Args:
        a (:obj:`NDArray`):
            First input array.
        b (:obj:`NDArray`):
            Second input array.

    Keyword args:
        k (:obj:`int`, optional):
            Diagonal above which to zero elements. `k = 0` (the default) is
            the main diagonal, `k < 0` is below it and `k > 0` is above.
        axis (:obj:`int`, optional):
            Array axis over which the outer product is summed. All by default.
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    if a.shape != b.shape:
        raise IndexError("shape of 'a' and 'b' does not match")
    # allocate output array
    dtype = (a[0] * b[0]).dtype  # correct dtype for product
    N = len(a)
    # sum all elements
    if axis is None:
        result = np.zeros_like(a[0], dtype=dtype)
        for i in range(min(N, N - k)):
            result += (a[i] * b[max(0, i + k) :]).sum(axis=0)
    # sum row-wise
    elif axis == 1:
        result = np.zeros_like(b, dtype=dtype)
        for i in range(min(N, N - k)):
            result[i] = (a[i] * b[max(0, i + k) :]).sum(axis=0)
    # sum column-wise
    elif axis == 0:
        result = np.zeros_like(a, dtype=dtype)
        for i in range(max(0, k), N):
            result[i] = (a[: min(N, max(0, i - k + 1))] * b[i]).sum(axis=0)
    return result[()]


def apply_bool_mask_ndim(
    array: _Tarr, mask: NDArray[np.bool_], axis: int | Sequence[int] | None = None
) -> _Tarr:
    """Apply a boolean mask (``mask``) to one or many axes of a numpy array."""
    if axis is None:
        axis = list(range(array.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    result = array
    for ax in axis:
        if result.shape[ax] != len(mask):
            raise IndexError(
                f"boolean index did not match indexed array along dimension "
                f"{ax}; dimension is {result.shape[ax]} but corresponding "
                f"boolean dimension is {len(mask)}"
            )
        result = np.compress(mask, result, axis=ax)
    return result


def apply_slice_ndim(
    array: _Tarr, item: int | slice | Sequence, axis: int | Sequence[int] | None = None
) -> _Tarr:
    """Apply an integer subset or slice (``item``) to one or many axes of a
    numpy array."""
    if axis is None:
        axis = list(range(array.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    if isinstance(item, slice):
        slices = [slice(None) for _ in range(array.ndim)]
        for ax in axis:
            slices[ax] = item
        return array[tuple(slices)]
    else:
        if isinstance(item, int):
            item = [item]
        indices = [range(n) for n in array.shape]
        for ax in axis:
            indices[ax] = item
        mesh_indices = np.ix_(*indices)
        return array[mesh_indices]


def sgn(val: ArrayLike) -> ArrayLike:
    """Compute the sign of a (array of) numbers, with positive numbers and 0
    returning 1, negative number returning -1."""
    return np.where(val == 0, 1.0, np.sign(val))


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    method: str,
    rowvar: bool = False,
    kind: str = "full",  # full, diag, var
) -> NDArray:
    """Compute a joint covariance from a sequence of data samples.

    These samples can be jackknife or bootstrap samples (etc.). If more than one
    set of samples is provided, the samples are concatenated along the second
    axis (default) or along the first axis if ``rowvar=True``.

    Args:
        samples (:obj`:NDArray`, :obj:`Sequence[NDArray]`):
            One or many sets of data samples. The number of samples must be
            identical.
        method (:obj:`str`, optional):
            The resampling method that generated the samples, see
            :obj:`~yaw.config.options.Options.method`.
        rowvar (:obj:`bool`, optional):
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind (:obj:`str`, optional):
            Determines the kind of covariance computed, see
            :obj:`~yaw.config.options.Options.kind`.
    """
    ax_samples = 1 if rowvar else 0
    ax_observ = 0 if rowvar else 1
    # if many samples are provided, concatenate them
    try:
        concat_samples = np.concatenate(samples, axis=ax_observ)
    except np.AxisError:
        concat_samples = samples

    # np.cov will produce a scalar value instead of matrix with shape (N,N)
    # for a single sample with shape (1,N)
    if concat_samples.shape[ax_samples] == 1:
        n_obs = concat_samples.shape[ax_observ]
        return np.full((n_obs, n_obs), np.nan)

    if method == "bootstrap":
        covmat = np.cov(concat_samples, rowvar=rowvar, ddof=1)
    elif method == "jackknife":
        n_samples = concat_samples.shape[ax_samples]
        covmat = np.cov(concat_samples, rowvar=rowvar, ddof=0) * (n_samples - 1)
    else:
        raise ValueError(f"invalid sampling method '{method}'")

    if kind == "full":
        pass
    elif kind == "diag":
        # get a matrix with only the main diagonal elements
        idx_diag = 0
        cov_diags = np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        try:
            for sample in samples:
                # go to next diagonal that contains correlations between samples
                idx_diag += sample.shape[ax_observ]
                # add just the diagonal values to the existing matrix
                cov_diags += np.diag(np.diag(covmat, k=-idx_diag), k=-idx_diag)
                cov_diags += np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        except IndexError:
            raise
        covmat = cov_diags
    elif kind == "var":
        covmat = np.diag(np.diag(covmat, k=0), k=0)
    else:
        raise ValueError(f"invalid covariance kind '{kind}'")
    return covmat


def global_covariance(
    data: Sequence[SampledData], method: str | None = None, kind: str = "full"
) -> NDArray:
    """Compute a joint covariance from a set of resampled data.

    Typically applied to a set of :obj:`CorrData`,
    :obj:`~yaw.redshifts.RedshiftData`, or :obj:`~yaw.redshifts.HistData`
    containers. The joint covariance is computed by concatenating the samples
    along the redshift binning axis.

    .. Warning::
        The input containers must have the same number of samples, and use the
        same resampling method. They also should be of the same type.

    Args:
        data (sequence of :obj:`SampledData`):
            The input containers, should be of the same type.
        method (:obj:`str`, optional):
            Specify the sampling method to use. All other containers must follow
            this convention.
        kind (:obj:`str`, optional):
            The method to compute the covariance matrix, see
            :func:`yaw.core.math.cov_from_samples`.

    Returns:
        :obj:`NDArray`:
            Jointly estimated covariance. Dimension matches the sum of all
            redshift bins in the input containers.
    """
    if len(data) == 0:
        raise ValueError("'data' must be a sequence with at least one item")
    if method is None:
        method = data[0].method
    for d in data[1:]:
        if d.method != method:
            raise ValueError("resampling method of data items is inconsistent")
    return cov_from_samples([d.samples for d in data], method=method, kind=kind)


def corr_from_cov(covariance: NDArray) -> NDArray:
    """Convert an input covariance matrix to a covariance matrix."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    return covariance / outer_v


def rebin(
    bins_new: NDArray[np.float_],
    bins_old: NDArray[np.float_],
    counts_old: NDArray[np.float_],
) -> NDArray[np.float_]:
    """Recompute compute histogram counts for a new binning.

    The new counts are computed by summing the fractional contribution of the
    counts from the original binning to the binning. The new binning may exceed
    or just partially cover the range of the original binning.

    Args:
        bins_new (:obj:`NDArray`):
            The new bin edges on which the counts are reevaluated.
        bins_old (:obj:`NDArray`):
            The bin edges from which the original counts were computed.
        counts_old (:obj:`NDArray`):
            The original histogram counts.

    Returns:
        :obj:`NDArray`:
            The histogram counts for the new binning.

    .. Note::
        Implemented as C extension.
    """
    return _rebin(
        bins_new.astype(np.float_),
        bins_old.astype(np.float_),
        counts_old.astype(np.float_),
    )


def shift_histogram(
    bins: NDArray, counts: NDArray, *, A: float = 1.0, dx: float = 0.0
) -> NDArray:
    """Shift a histogram by a fixed value.

    The histogram values are recomputed for new bin edges that are shifted by
    the desired amount using :func:`rebin`. The normalisation of the histogram
    may change if the shifted binning does not cover the full range of the data.

    Args:
        bins (:obj:`NDArray`):
            The bin edges from which the counts were computed.
        counts_old (:obj:`NDArray`):
            The histogram counts.

    Keyword args:
        A (:obj:`float`, optional):
            Scalar amplitude used to rescale the new histgram counts.
        dx (:obj:`float`, optional):
            Amount by which the histogram (i.e. the bin edges) are shifted.

    Returns:
        :obj:`NDArray`:
            The shifted histogram counts.
    """
    bins_old = bins.astype(np.float_)
    bins_new = bins_old + dx
    return A * _rebin(bins_new, bins_old, counts.astype(np.float_))
