from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import h5py
import numpy as np
import pandas as pd
import scipy.sparse

from yaw.core.config import ResamplingConfig
from yaw.core.datapacks import SampledData
from yaw.core.utils import BinnedQuantity, HDFSerializable, PatchedQuantity

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from numpy.typing import ArrayLike, NDArray, DTypeLike
    from pandas import Interval, IntervalIndex
    from treecorr import NNCorrelation
    from yaw.core.utils import TypePatchKey


logger = logging.getLogger(__name__.replace(".core.", "."))


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


TypeSlice: TypeAlias = Union[slice, int, None]


class PatchedArray(BinnedQuantity, PatchedQuantity, HDFSerializable):

    density: float

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        shape = self.shape
        density = self.density
        return f"{string}, {shape=}, {density=})"

    def _parse_key(
        self,
        key: tuple | TypeSlice
    ) -> tuple[TypeSlice, TypeSlice, TypeSlice]:
        default = slice(None, None, None)
        j, k = default, default
        if not isinstance(key, tuple):
            i = key
        else:
            if len(key) == 2:
                i, j = key
            elif len(key) == 3:
                i, j, k = key
            else:
                raise IndexError(
                    f"too many indices for array: array is 3-dimensional, but "
                    f"{len(key)} were indexed")
        return i, j, k

    def __getitem__(self, key) -> ArrayLike:
        raise NotImplementedError

    @property
    def dtype(self) -> DTypeLike:
        return np.float_

    @property
    def shape(self) -> tuple[int]:
        return (self.n_patches, self.n_patches, self.n_bins)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def as_array(self) -> NDArray:
        return self[:, :, :]

    def get_sum(self, config: ResamplingConfig) -> SampledData:
        if config.method == "bootstrap":
            data = self._sum(config.crosspatch)
            samples = self._sum_bootstrap(config)
        else:
            data, samples = self._sum_jackknife(config)
        return SampledData(
            binning=self.binning,
            data=data,
            samples=samples,
            method=config.method)


def binning_from_hdf(source: h5py.Group) -> IntervalIndex:
    dset = source["binning"]
    left, right = dset[:].T
    closed = dset.attrs["closed"]
    return pd.IntervalIndex.from_arrays(left, right, closed=closed)


def binning_to_hdf(binning: IntervalIndex, dest: h5py.Group) -> None:
    edges = np.column_stack([binning.left, binning.right])
    dset = dest.create_dataset("binning", data=edges, **_compression)
    dset.attrs["closed"] = binning.closed


class PatchedTotal(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        totals1: NDArray,
        totals2: NDArray,
        *,
        auto: bool
    ) -> None:
        self.binning = binning
        for i, totals in enumerate((totals1, totals2), 1):
            if totals.ndim != 2:
                raise ValueError(f"'totals{i}' must be two dimensional")
            if totals.shape[1] != self.n_bins:
                raise ValueError(
                    f"number of bins for 'totals{i}' does not match 'binning'")
        if totals1.shape != totals2.shape:
            raise ValueError(
                f"number of patches and bins do not match: "
                f"{totals1.shape} != {totals2.shape}")
        self.totals1 = totals1
        self.totals2 = totals2
        self.auto = auto

    def __getitem__(self, key) -> ArrayLike:
        i, j, k = self._parse_key(key)
        x = np.atleast_2d(self.t1[i])[:, k]
        y = np.atleast_2d(self.t2[j])[:, k]
        arr = np.einsum("i...,j...->ij...", x, y)
        squeeze_ax = tuple(
            a for a, val in enumerate((i, j))
            if isinstance(val, (int, np.integer)))
        return np.squeeze(arr, axis=squeeze_ax)

    @property
    def n_patches(self) -> int:
        return self.totals1.shape[0]

    @property
    def density(self) -> float:
        return (self.totals1.size + self.totals2.size) / self.size

    def _sum_jackknife(
        self,
        config:ResamplingConfig
    ) -> tuple[NDArray, NDArray]:
        samples = np.empty((self.n_patches, self.n_bins))

        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        half_diag = diag * 0.5

        if not config.crosspatch:  # only diagonal terms, leave out element k
            if self.auto:
                diag = half_diag
            full = np.einsum("i...->...", diag)
            for k in range(self.n_patches):
                samples[k] = full - diag[k]

        else:
            # for the k-th jackknife sample:
            #     full - cols[k] - rows[k] + diag[k]
            rows = np.einsum("i...,j...->i...", self.totals1, self.totals2)

            if self.auto:
                # trick to compute upper triangle of
                # np.einsum("i...,j...->..." self.totals1, self.totals2)
                idx_row, idx_col = np.triu_indices(self.n_patches, 0)
                full = np.einsum(
                    "i...,i...", self.totals1[idx_row], self.totals2[idx_col])
                # NOTE: full and rows both contain the full diagonal
                full -= half_diag.sum()
                rows = rows - half_diag
                for k in range(self.n_patches):
                    # for upper triangle, row contains rows[k], cols[k], diag[k]
                    samples[k] = full - rows[k]
            else:
                # sum along rows
                cols = np.einsum("i...,j...->j...", self.totals1, self.totals2)
                full = np.einsum("i...->...", rows)
                for k in range(self.n_patches):
                    # subtracting row and colum subtracts diagonal twice
                    samples[k] = full - cols[k] - rows[k] + diag[k]
       
        return full, samples

    def _sum(self, crosspatch: bool) -> tuple[NDArray, NDArray]:
        raise NotImplementedError
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        if self.auto:
            diag *= 2.0

        if not crosspatch:  # only diagonal terms
            full = np.einsum("i...->...", diag)
        elif not self.auto:
            full = np.einsum("i...,j...->...", self.totals1, self.totals2)
        else:
            # trick to compute upper triangle of case above
            idx_row, idx_col = np.triu_indices(self.n_patches, 0)
            full = np.einsum(
                "i...,i...", self.totals1[idx_row], self.totals2[idx_col])
        return full

    def _single_bootstrap(
        self,
        patch_idx: NDArray[np.int_],
        config: ResamplingConfig
    ) -> NDArray:
        raise NotImplementedError
        # create the realisation
        tot1_real = self.totals1[patch_idx]
        tot2_real = self.totals2[patch_idx]

        diag = np.einsum("i...,i...->...", tot1_real, tot2_real)
        if self.auto:
            diag *= 2.0

        if not config.crosspatch:  # only diagonal terms
            return diag

        # need to skip autocorrelation terms that arise on off-diagonals due to
        # repeated elements in realisation: identify where in the matrix the
        # realisation indices are not identical
        is_cross = np.subtract.outer(patch_idx, patch_idx) != 0
        if self.auto:
            is_cross = np.triu(is_cross)
        # compute the sum over the outer product, setting auto-terms to zero
        cross = np.einsum("i...,j...,ij->...", tot1_real, tot2_real, is_cross)
        return diag + cross

    def _sum_bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError
        out = np.empty((config.n_boot, self.n_bins))  # bootstrap samples
        # create and iterate realisations
        patch_idx = config.get_samples(self.n_patches)
        for n, idx in enumerate(patch_idx):
            out[n] = self._single_bootstrap(idx, config=config)

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PatchedTotal:
        # reconstruct the binning
        binning = binning_from_hdf(source)
        # load the data
        totals1 = source["totals1"][:]
        totals2 = source["totals2"][:]
        auto = source["auto"][()]
        return cls(
            binning=binning,
            totals1=totals1,
            totals2=totals2,
            auto=auto)

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.binning, dest)
        # store the data
        dest.create_dataset("totals1", data=self.totals1, **_compression)
        dest.create_dataset("totals2", data=self.totals2, **_compression)
        dest.create_dataset("auto", data=self.auto)


class PatchedCount(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        n_patches: int,
        *,
        auto: bool,
        dtype: DTypeLike = np.float_
    ) -> None:
        self.binning = binning
        self._keys = set()
        self._n_patches = n_patches
        self._bins: list[spmatrix] = [
            scipy.sparse.dok_matrix((n_patches, n_patches), dtype=dtype)
            for i in range(self.n_bins)]
        self.auto = auto

    def __getitem__(self, key) -> ArrayLike:
        i, j, k = self._parse_key(key)
        squeeze_ax = tuple(
            a for a, val in enumerate((i, j, k))
            if isinstance(val, (int, np.integer)))
        arr = np.array([counts[i, j].toarray() for counts in self._bins[k]])
        return np.squeeze(arr, axis=squeeze_ax)

    def __setitem__(self, key: TypePatchKey, item: NDArray):
        item = np.asarray(item)
        if item.shape != (self.n_bins,):
            raise ValueError(
                f"can only set items with length 'n_bins'={self.n_bins}")
        if not isinstance(key, tuple):
            raise TypeAlias(f"slice must be of type {tuple}")
        elif len(key) != 2:
            raise IndexError(
                f"too many indices for array assignment: index must be "
                f"2-dimensional, but {len(key)} where indexed")
        for n, val in enumerate(key):
            if not isinstance(val, (int, np.integer)):
                raise TypeError(
                    f"index for axis {n} must be of type {int}, "
                    f"but got {type(val)}")
        for counts, val in zip(self._bins, item):
            counts[key] = val
        self._keys.add(key)

    def keys(self) -> NDArray:
        return np.array(list(self._keys))

    def values(self) -> NDArray:
        idx_ax0, idx_ax1 = self.keys().T
        values = np.column_stack([
            np.squeeze(counts[idx_ax0, idx_ax1].toarray())
            for counts in self._bins])
        return values

    @property
    def n_patches(self) -> int:
        return self._n_patches

    @property
    def n_bins(self) -> int:
        return len(self.binning)

    @property
    def density(self) -> float:
        stored = sum(counts.nnz for counts in self._bins)
        total = np.prod(self.shape)
        return stored / total

    def _bin_jackknife(
        self,
        counts: spmatrix,
        config: ResamplingConfig
    ) -> tuple[np.float_, NDArray]:
        samples = np.empty(self.n_patches)

        diag = counts.diagonal()
        if self.auto:
            diag *= 0.5

        if not config.crosspatch:  # only diagonal terms, leave out element k
            full = diag.sum()
            for k in range(self.n_patches):
                samples[k] = full - diag[k]

        else:
            # for the k-th jackknife sample:
            #     full - cols[k] - rows[k] + diag[k]
            if self.auto:
                counts = scipy.sparse.triu(counts)
            # sum along columns
            rows = counts.sum(axis=1)
            # sum along rows
            cols = counts.sum(axis=0)
            # operations above result in numpy.matrix, squeeze extra dimension
            rows = np.squeeze(np.asarray(rows))
            cols = np.squeeze(np.asarray(cols))
            full = rows.sum()

            if self.auto:
                # NOTE: full and rows/cols both contain the full diagonal but
                # we need to scale this by two
                diag_scaled = diag.sum()
                full -= diag_scaled
                rows = rows - diag_scaled
                cols = cols - diag_scaled

            for k in range(self.n_patches):
                # subtracting row and colum subtracts diagonal twice
                samples[k] = full - cols[k] - rows[k] + diag[k]

        return full, samples

    def _sum_jackknife(
        self,
        config: ResamplingConfig
    ) -> tuple[NDArray, NDArray]:
        data = np.empty(self.n_bins)
        samples = np.empty((self.n_patches, self.n_bins))
        for i, counts in enumerate(self._bins):
            dat, samp = self._bin_jackknife(counts, config=config)
            data[i] = dat
            samples[:, i] = samp
        return data, samples

    def _bin_sum(
        self,
        counts: spmatrix,
        config: ResamplingConfig
    ) -> NDArray:
        raise NotImplementedError
        diag = counts.diagonal()
        if self.auto:
            diag *= 2.0

        if not config.crosspatch:  # only diagonal terms
            full = diag.sum()

        if self.auto:
            counts = scipy.sparse.triu(counts)

        return counts.sum()

    def _bin_single_bootstrap(
        self,
        counts: np.matrix,
        patch_idx: NDArray[np.int_],
        config: ResamplingConfig
    ) -> np.float_:
        raise NotImplementedError
        diag = counts.diagonal()[patch_idx].sum()  # diagonal of realisation
        if self.auto:
            diag *= 2.0

        if not config.crosspatch:  # only diagonal terms
            return diag

        # create realisation of counts matrix
        counts_real = counts[patch_idx][:, patch_idx]
        # need to skip autocorrelation terms that araise on off-diagonal due to
        # repeated elements in realisation
        # identify where in the matrix the realisation indices not identical
        is_cross = np.subtract.outer(patch_idx, patch_idx) != 0
        if self.auto:
            is_cross = np.triu(is_cross)

        # compute the sum over the outer product, setting auto-terms to zero
        cross = counts_real[is_cross].sum()
        return diag + cross  # scalar!

    def _bin_bootstrap(
        self,
        counts: spmatrix,
        config: ResamplingConfig
    ) -> NDArray:
        raise NotImplementedError
        out = np.empty(config.n_boot)
        # indexing in both dimensions very inefficient in sparse matrices,
        # switch to dense numpy array
        counts: np.matrix = counts.toarray()
        # create and iterate realisations
        patch_idx = config.get_samples(self.n_patches)
        for n, idx in enumerate(patch_idx):
            out[n] = self._bin_single_bootstrap(counts, idx, config=config)

    def _sum_bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError
        out = np.empty((config.n_boot, self.n_bins))
        for i, counts in enumerate(self._bins):
            bin_jackknife = self._bin_bootstrap(counts, config=config)
            out[:, i] = bin_jackknife
        return out

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PatchedTotal:
        # reconstruct the binning
        binning = binning_from_hdf(source)
        # load the data
        keys = [tuple(key) for key in source["keys"][:]]
        data = source["data"][:]
        n_patches = source["n_patches"][()]
        auto = source["auto"][()]
        # reconstruct the sparse matrix incrementally
        new = cls(
            binning=binning, n_patches=n_patches, auto=auto, dtype=data.dtype)
        for key, value in zip(keys, data):
            new[key] = value
        return new

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.binning, dest)
        # store the data
        dest.create_dataset("keys", data=self.keys(), **_compression)
        dest.create_dataset("data", data=self.values(), **_compression)
        dest.create_dataset("n_patches", data=self.n_patches)
        dest.create_dataset("auto", data=self.auto)


@dataclass(frozen=True)
class PairCountResult(PatchedQuantity, BinnedQuantity, HDFSerializable):

    count: PatchedCount
    total: PatchedTotal

    def __post_init__(self) -> None:
        if self.count.n_patches != self.total.n_patches:
            raise ValueError(
                "number of patches of 'count' and total' do not match")
        if self.count.n_bins != self.total.n_bins:
            raise ValueError(
                "number of bins of 'count' and total' do not match")

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: NNCorrelation
    ) -> PairCountResult:
        raise NotImplementedError
        # extract the (cross-patch) pair counts
        n_patches = max(correlation.npatch1, correlation.npatch2)
        
        keys = []
        count = np.empty((len(correlation.results), 1))
        total = np.empty((len(correlation.results), 1))
        for i, (patches, result) in enumerate(correlation.results.items()):
            keys.append(patches)
            count[i] = result.weight
            total[i] = result.tot
        return cls(
            n_patches=n_patches,
            count=PatchedCount(keys, count),
            total=PatchedTotal(keys, total),
            mask=correlation._ok,
            binning=pd.IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[PairCountResult]
    ) -> PairCountResult:
        raise NotImplementedError
        # check that the data is compatible
        if len(zbins) == 0:
            raise ValueError("'zbins' is empty")
        n_patches = zbins[0].n_patches
        mask = zbins[0].mask
        keys = tuple(zbins[0].keys())
        nbins = len(zbins[0])
        for zbin in zbins[1:]:
            if zbin.n_patches != n_patches:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
            if tuple(zbin.keys()) != keys:
                raise ValueError("patches are inconsistent")
            if len(zbin) != nbins:
                raise IndexError("number of bins is inconsistent")

        # check the ordering of the bins based on the provided intervals
        binning = pd.IntervalIndex.from_tuples([
            zbin.binning.to_tuples()[0]  # contains just one entry
            for zbin in zbins])
        if not binning.is_non_overlapping_monotonic:
            raise ValueError(
                "the binning interval is overlapping or not monotonic")
        for this, following in zip(binning[:-1], binning[1:]):
            if this.right != following.left:
                raise ValueError(f"the binning interval is not contiguous")

        # merge the ArrayDicts
        """
        count = ArrayDict(
            keys, np.column_stack([zbin.count.as_array() for zbin in zbins]))
        total = ArrayDict(
            keys, np.column_stack([zbin.total.as_array() for zbin in zbins]))
        return cls(
            n_patches=n_patches,
            count=count,
            total=total,
            mask=mask,
            binning=binning)
        """

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        n_keys = len(self.keys())
        return f"{string}, {n_patches=}, {n_keys=})"

    @property
    def binning(self) -> IntervalIndex:
        return self.total.binning

    @property
    def n_patches(self) -> int:
        return self.total.n_patches

    def get(self, config: ResamplingConfig) -> SampledData:
        counts = self.count.get_sum(config)
        totals = self.total.get_sum(config)
        return SampledData(
            binning=self.binning,
            data=(counts.data / totals.data),
            samples=(counts.samples / totals.samples),
            method=config.method)

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PairCountResult:
        count = PatchedCount.from_hdf(source["count"])
        total = PatchedTotal.from_hdf(source["total"])
        return cls(count=count, total=total)

    def to_hdf(self, dest: h5py.Group) -> None:
        group = dest.create_group("count")
        self.count.to_hdf(group)
        group = dest.create_group("total")
        self.total.to_hdf(group)
