from __future__ import annotations

import itertools
from collections.abc import Collection, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd

from yaw.core.utils import (
    BinnedQuantity, HDFSerializable, PatchedQuantity, TypePathStr)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval, IntervalIndex
    from treecorr import NNCorrelation
    from yaw.core.utils import TypePatchKey


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


class ArrayDict(Mapping):

    def __init__(
        self,
        keys: Collection[Any],
        array: NDArray
    ) -> None:
        if len(array) != len(keys):
            raise ValueError("number of keys and array length do not match")
        self._array = array
        self._dict = {key: idx for idx, key in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, key: Any) -> NDArray:
        idx = self._dict[key]
        return self._array[idx]

    def __iter__(self) -> Iterator[NDArray]:
        return self._dict.__iter__()

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def items(self) -> list[tuple[Any, NDArray]]:
        # ensure that the items are ordered by the index of each key
        return sorted(self._dict.items(), key=lambda item: item[1])

    def keys(self) -> list[Any]:
        # key are ordered by their corresponding index
        return [key for key, _ in self.items()]

    def values(self) -> list[NDArray]:
        # values are returned in index order
        return [value for value in self._array]

    def get(self, key: Any, default: Any) -> Any:
        try:
            idx = self._dict[key]
        except KeyError:
            return default
        else:
            return self._array[idx]

    def sample(self, keys: Iterable[Any]) -> NDArray:
        idx = [self._dict[key] for key in keys]
        return self._array[idx]

    def as_array(self) -> NDArray:
        return self._array

    def as_dataframe(self) -> DataFrame:
        return pd.DataFrame(self.as_array(), index=self.keys())


def arraydict_to_hdf(array: ArrayDict, dest: h5py.Group) -> None:
    # we use tuple keys, pairs of ints
    keys = np.array(array.keys())
    dest.create_dataset("keys", data=keys, **_compression)
    dest.create_dataset("data", data=array._array, **_compression)


def arraydict_from_hdf(source: h5py.Group) -> ArrayDict:
    # we use tuple keys, pairs of ints
    keys = [tuple(key) for key in source["keys"][:]]
    data = source["data"][:]
    return ArrayDict(keys, data)


def bootstrap_iter(
    index: NDArray[np.int_],
    mask: NDArray[np.bool_]
) -> Generator[TypePatchKey, None, None]:
    """from TreeCorr.BinnedCorr2"""
    # Include all represented auto-correlations once, repeating as appropriate.
    # This needs to be done separately from the below step to avoid extra
    # pairs (i,i) that you would get by looping i in index and j in index for
    # cases where i=j at different places in the index list.  E.g. if i=3 shows
    # up 3 times in index, then the naive way would get 9 instance of (3,3),
    # whereas we only want 3 instances.
    ret1 = ((i, i) for i in index if mask[i, i])
    # And all other pairs that aren't really auto-correlations.
    # These can happen at their natural multiplicity from i and j loops.
    ret2 = ((i, j) for i in index for j in index if mask[i, j] and i != j)
    # merge generators
    return itertools.chain(ret1, ret2)


def jackknife_iter(
    patch_key_list: Iterable[TypePatchKey],
    drop_index: int,
    mask: NDArray[np.bool_]
) -> Generator[TypePatchKey, None, None]:
    """from TreeCorr.BinnedCorr2"""
    return ((j, k) for j, k in patch_key_list
            if j != drop_index and k != drop_index and mask[j, k])


@dataclass(frozen=True)
class PairCountResult(PatchedQuantity, BinnedQuantity, HDFSerializable):

    count: ArrayDict
    total: ArrayDict
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
        seed: int = 12345
    ) -> NDArray[np.int_]:
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, self.n_patches, size=(n_boot, self.n_patches))

    def _get_jackknife_samples(
        self,
        global_norm: bool = False,
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
        global_norm: bool = False
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
        method: str = "bootstrap",
        n_boot: int = 500,
        patch_idx: NDArray[np.int_] | None = None,
        global_norm: bool = False,
        seed: int = 12345
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
