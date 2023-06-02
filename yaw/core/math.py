from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numba
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike


_Tarr = TypeVar("_Tarr", bound=NDArray)


def array_equal(arr1: NDArray, arr2: NDArray) -> bool:
    return (
        isinstance(arr1, np.ndarray) and
        isinstance(arr2, np.ndarray) and
        arr1.shape == arr2.shape and
        (arr1 == arr2).all())


def outer_triu_sum(
    a: ArrayLike,
    b: ArrayLike,
    *,
    k: int = 0,
    axis: int | None = None
) -> NDArray:
    """
    Equivalent to
        np.triu(np.outer(a, b), k).sum(axis)
    but supports extra dimensions in a and b and does not construct the full
    outer product matrix.
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
        for i in range(min(N, N-k)):
            result += (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum row-wise
    elif axis == 1:
        result = np.zeros_like(b, dtype=dtype)
        for i in range(min(N, N-k)):
            result[i] = (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum column-wise
    elif axis == 0:
        result = np.zeros_like(a, dtype=dtype)
        for i in range(max(0, k), N):
            result[i] = (a[:min(N, max(0, i-k+1))] * b[i]).sum(axis=0)
    return result[()]


def apply_bool_mask_ndim(
    array: _Tarr,
    mask: NDArray[np.bool_],
    axis: int | Sequence[int] | None = None
) -> _Tarr:
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
                f"boolean dimension is {len(mask)}")
        result = np.compress(mask, result, axis=ax)
    return result


def apply_slice_ndim(
    array: _Tarr,
    item: int | slice | Sequence,
    axis: int | Sequence[int] | None = None
) -> _Tarr:
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
    return np.where(val == 0, 1.0, np.sign(val))


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    method: str,
    rowvar: bool = False,
    kind: str = "full"  # full, diag, var
) -> NDArray:
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


def corr_from_cov(covariance: NDArray) -> NDArray:
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    return covariance / outer_v


@numba.njit
def rebin(
    bins_new: NDArray,
    bins_old: NDArray,
    counts_old: NDArray
) -> NDArray:
    n_bins_old = len(bins_old) - 1
    n_bins_new = len(bins_new) - 1

    # ensure numpy
    counts_old = np.asarray(counts_old)
    counts_new = np.zeros(n_bins_new, dtype=np.float_)

    # iterate the new bins and check which of the old bins overlap with it
    for i_new in range(n_bins_new):
        zmin_n = bins_new[i_new]
        zmax_n = bins_new[i_new+1]

        for i_old in range(n_bins_old):
            zmin_o = bins_old[i_old]
            zmax_o = bins_old[i_old+1]
            count = counts_old[i_old]

            # check for full or partial overlap
            contains = (zmin_n >= zmin_o) & (zmax_n < zmax_o)
            overlaps_min = (zmin_n <= zmin_o) & (zmax_n > zmin_o)
            overlaps_max = (zmin_n <= zmax_o) & (zmax_n > zmax_o)

            if contains | overlaps_min | overlaps_max:
                # compute fractional bin overlap 
                zmin_overlap = max(zmin_o, zmin_n)
                zmax_overlap = min(zmax_o, zmax_n)
                fraction = (zmax_overlap - zmin_overlap) / (zmax_o - zmin_o)

                # assume uniform distribution of data in bin and increment
                # counts by the bin count weighted by the overlap fraction
                counts_new[i_new] += count * fraction

    return counts_new


def shift_histogram(
    bins: NDArray,
    counts: NDArray,
    *,
    A: float = 1.0,
    dx: float = 0.0
) -> NDArray:
    return A * rebin(bins+dx, bins, counts)


def round_to(value: int, to: int) -> int:
    return int(np.ceil(value / to) * to)
