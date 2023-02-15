from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from yaw.core import default as DEFAULT
from yaw.core.utils import BinnedQuantity, HDFSerializable, PatchedQuantity

if TYPE_CHECKING:
    import numba
    from numpy.typing import NDArray, DTypeLike
    from pandas import DataFrame, Interval, IntervalIndex
    from treecorr import NNCorrelation
    from yaw.core.utils import TypePatchKey

logger = logging.getLogger(__name__.replace(".core.", "."))
try:
    from numba import njit
except Exception:
    logger.warn("numba not available, resampling performance degraded")
    from functools import wraps

    def njit(func):  # TODO: probably need optional dummy decorator arguments
        @wraps
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


class PatchDict(HDFSerializable):

    @staticmethod
    def _check_key(key: TypePatchKey) -> None:
        if not isinstance(key, tuple):
            raise TypeError(f"keys must be of type {tuple}")
        elif len(key) != 2:
            raise ValueError(f"keys must be tuples of length 2, got {len(key)}")


class CountsDict(PatchDict, dict):
    
    _default_item: NDArray = None

    def __init__(self, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        # need to do this
        for key, item in self.items():
            self._check_key(key)
            self._check_item(item)

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def _check_item(self, item: NDArray) -> None:
        if not isinstance(item, np.ndarray):
            raise TypeError(f"values must be of type {np.ndarray}")
        if self._default_item is None:
            self._default_item = np.zeros_like(item)
        elif item.shape != self._default_item.shape:
            raise ValueError(
                f"expected value array of shape {self._default_item.shape}, "
                f"got {item.shape}")
        elif item.dtype != self._default_item.dtype:
            raise TypeError(
                f"value arrays must be of type {self._default_item.dtype}, "
                f"got {item.dtype}")

    def __getitem__(self, key: TypePatchKey) -> NDArray:
        return super().__getitem__(key)

    def __setitem__(self, key: TypePatchKey, item: NDArray):
        self._check_key(key)
        self._check_item(item)
        super().__setitem__(key, item)
    
    def get(self, item):
        return super().get(item, self._default_item)

    def as_numba_dict(self) -> numba.typed.Dict | CountsDict:
        try:
            import numba
        except Exception:
            the_dict = self
        else:
            the_dict = numba.typed.Dict()
            for key, value in self.items():
                the_dict[key] = value
        return the_dict

    def as_array(self) -> NDArray:
        array = np.empty(
            (len(self), *self._default_item.shape),
            dtype=self._default_item.dtype)
        for i, value in enumerate(self.values()):
            array[i] = value
        return array

    def as_dataframe(self) -> DataFrame:
        return pd.DataFrame(self.as_array(), index=list(self.keys()))

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> CountsDict:
        # we use tuple keys, pairs of ints
        keys = [tuple(key) for key in source["keys"][:]]
        data = source["data"][:]
        result = cls()
        for key, value in zip(key, data):
            result[key] = value
        return result

    def to_hdf(self, dest: h5py.Group) -> None:
        # we use tuple keys, pairs of ints
        keys = np.array(list(self.keys()))
        dest.create_dataset("keys", data=keys, **_compression)
        dest.create_dataset("data", data=self.as_array(), **_compression)


class TotalsDict(PatchDict, Mapping):

    def __init__(self, totals1: NDArray, totals2: NDArray) -> None:
        if totals1.shape != totals2.shape:
            raise ValueError(
                f"number of patches and bins do not match: "
                f"{totals1.shape} != {totals2.shape}")
        self.totals1 = totals1
        self.totals2 = totals2

    def __getitem__(self, key: TypePatchKey) -> int | float:
        self._check_key(key)
        k1, k2 = key
        try:
            return self.totals1[k1] * self.totals2[k2]
        except IndexError as e:
            raise KeyError(key) from e


@numba.njit
def accumulate_counts_realisation(
    index: NDArray[np.int_],
    mask: NDArray[np.bool_],
    thedict: numba.typed.Dict,
    o_shape: tuple,
    o_dtype: DTypeLike
) -> NDArray:
    """from TreeCorr.BinnedCorr2, works for jackknife and bootstrap"""
    total = np.zeros(o_shape, dtype=o_dtype)
    # Include all represented auto-correlations once, repeating as appropriate.
    # This needs to be done separately from the below step to avoid extra
    # pairs (i,i) that you would get by looping i in index and j in index for
    # cases where i=j at different places in the index list.  E.g. if i=3 shows
    # up 3 times in index, then the naive way would get 9 instance of (3,3),
    # whereas we only want 3 instances.
    for i in index:
        key = (i, i)
        if mask[i, i]:  # TODO: need mask?
            if key in thedict:
                total += thedict[key]
    # And all other pairs that aren't really auto-correlations.
    # These can happen at their natural multiplicity from i and j loops.
    for i in index:
        for j in index:
            if i != j and mask[i, j]:  # TODO: need mask?
                key = (i, j)
                if key in thedict:
                    total += thedict[key]
    return total


@numba.njit
def accumulate_totals_realisation(
    index: NDArray[np.int_],
    mask: NDArray[np.bool_],
    totals1: NDArray,
    totals2: NDArray
) -> NDArray:
    """from TreeCorr.BinnedCorr2, works for jackknife and bootstrap"""
    total = np.zeros(totals1.shape[1], dtype=totals1.dtype)
    # Include all represented auto-correlations once, repeating as appropriate.
    # This needs to be done separately from the below step to avoid extra
    # pairs (i,i) that you would get by looping i in index and j in index for
    # cases where i=j at different places in the index list.  E.g. if i=3 shows
    # up 3 times in index, then the naive way would get 9 instance of (3,3),
    # whereas we only want 3 instances.
    for i in index:
        if mask[i, i]:  # TODO: need mask?
            total += totals1[i] * totals2[i]
    # And all other pairs that aren't really auto-correlations.
    # These can happen at their natural multiplicity from i and j loops.
    for i in index:
        for j in index:
            if i != j and mask[i, j]:  # TODO: need mask?
                total += totals1[i] * totals2[j]
    return total


@dataclass(frozen=True)
class PairCountResult(PatchedQuantity, BinnedQuantity, HDFSerializable):

    count: CountsDict
    total: CountsDict
    mask: NDArray[np.bool_]
    binning: IntervalIndex
    n_patches: int

    def __post_init__(self) -> None:
        if self.count.keys() != self.total.keys():
            raise KeyError("keys for 'count' and 'total' are not identical")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        n_keys = len(self.keys())
        return f"{string}, {n_patches=}, {n_keys=})"

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: NNCorrelation
    ) -> PairCountResult:
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
            count=ArrayDict(keys, count),
            total=ArrayDict(keys, total),
            mask=correlation._ok,
            binning=pd.IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[PairCountResult]
    ) -> PairCountResult:
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

    def keys(self) -> list[TypePatchKey]:
        return self.total.keys()

    def get(self) -> PairCountData:
        return PairCountData(
            count=self.count.as_array().sum(axis=0),
            total=self.total.as_array().sum(axis=0),
            binning=self.binning)

    def _generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = DEFAULT.Resampling.seed
    ) -> NDArray[np.int_]:
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, self.n_patches, size=(n_boot, self.n_patches))

    def _get_jackknife_samples(
        self,
        global_norm: bool = DEFAULT.Resampling.global_norm,
    ) -> PairCountData:
        # The iterator expects a single patch index which is dropped in a single
        # realisation.
        count = []
        total = []
        if global_norm:
            global_total = self.total.as_array().sum(axis=0)
        for idx in range(self.n_patches):  # leave-one-out iteration
            # we need to use the jackknife iterator twice
            patches = list(jackknife_iter(self.keys(), idx, mask=self.mask))
            idx = [self.count._dict[key] for key in patches]  # avoid repeating
            count.append(self.count._array[idx].sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total._array[idx].sum(axis=0))
        return PairCountData(
            count=np.array(count),
            total=np.array(total),
            binning=self.binning)

    def _get_bootstrap_samples(
        self,
        patch_idx: NDArray[np.int_],
        global_norm: bool = DEFAULT.Resampling.global_norm
    ) -> PairCountData:
        # The treecorr bootstrap iterator expects a list of patch indicies which
        # are present in the specific boostrap realisation to generate, i.e.
        # draw N times from (0, ..., N) with repetition. These random patch
        # indices for M realisations should be provided in the [M, N] shaped
        # array 'patch_idx'.
        count = []
        total = []
        if global_norm:
            global_total = self.total.as_array().sum(axis=0)
        for idx in patch_idx:  # simplified leave-one-out iteration
            # we need to use the jackknife iterator twice
            patches = list(bootstrap_iter(idx, mask=self.mask))
            idx = [self.count._dict[key] for key in patches]  # avoid repeating
            count.append(self.count._array[idx].sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total._array[idx].sum(axis=0))
        return PairCountData(
            count=np.array(count),
            total=np.array(total),
            binning=self.binning)

    def get_samples(
        self,
        *,
        method: str = DEFAULT.Resampling.method,
        n_boot: int = DEFAULT.Resampling.n_boot,
        patch_idx: NDArray[np.int_] | None = None,
        global_norm: bool = DEFAULT.Resampling.global_norm,
        seed: int = DEFAULT.Resampling.seed
    ) -> PairCountData:
        if method == "jackknife":
            samples = self._get_jackknife_samples(global_norm=global_norm)
        elif method == "bootstrap":
            if patch_idx is None:
                patch_idx = self._generate_bootstrap_patch_indices(
                    n_boot, seed=seed)
            samples = self._get_bootstrap_samples(
                patch_idx, global_norm=global_norm)
        else:
            raise NotImplementedError(
                f"sampling method '{method}' not implemented")
        return samples

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PairCountResult:
        n_patches = source["n_patches"][()]
        mask = source["mask"][:]
        count = arraydict_from_hdf(source["count"])
        total = arraydict_from_hdf(source["total"])
        dset = source["binning"]
        left, right = dset[:].T
        binning = pd.IntervalIndex.from_arrays(
            left, right, closed=dset.attrs["closed"])
        return cls(
            count=count, total=total, mask=mask, binning=binning,
            n_patches=n_patches)

    def to_hdf(self, dest: h5py.Group) -> None:
        dest.create_dataset("n_patches", data=self.n_patches)  # scalar
        arraydict_to_hdf(self.count, dest.create_group("count"))
        arraydict_to_hdf(self.total, dest.create_group("total"))
        dest.create_dataset("mask", data=self.mask, **_compression)
        binning = np.column_stack([
            self.binning.left.values, self.binning.right.values])
        dset = dest.create_dataset(
            "binning", data=binning, **_compression)
        dset.attrs["closed"] = self.binning.closed


@dataclass(frozen=True, repr=False)
class PairCountData(BinnedQuantity):

    count: NDArray[np.float_]
    total: NDArray[np.float_]
    binning: IntervalIndex

    def is_compatible(self, other: PairCountData) -> bool:
        return super().is_compatible(other)

    def normalise(self) -> NDArray[np.float_]:
        normalised = self.count / self.total
        return pd.DataFrame(data=normalised.T, index=self.binning)
