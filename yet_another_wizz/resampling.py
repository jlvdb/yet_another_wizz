from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, IntervalIndex

from yet_another_wizz.utils import ArrayDict, TypePatchKey


@dataclass(frozen=True, repr=False)
class PairCountData:
    binning: IntervalIndex
    count: NDArray[np.float_]
    total: NDArray[np.float_]

    def normalise(self) -> NDArray[np.float_]:
        normalised = self.count / self.total
        return DataFrame(data=normalised.T, index=self.binning)


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


@dataclass(frozen=True, repr=False)
class PairCountResult:
    npatch: int
    count: ArrayDict
    total: ArrayDict
    mask: NDArray[np.bool_]
    binning: IntervalIndex

    def __post_init__(self) -> None:
        if self.count.keys() != self.total.keys():
            raise KeyError("keys for 'count' and 'total' are not identical")

    def keys(self) -> list[TypePatchKey]:
        return self.total.keys()

    @property
    def nbins(self) -> int:
        return len(self.binning)

    def get_patch_count(self) -> DataFrame:
        return DataFrame(
            index=self.binning,
            columns=self.keys(),
            data=self.count.as_array().T)

    def get_patch_total(self) -> DataFrame:
        return DataFrame(
            index=self.binning,
            columns=self.keys(),
            data=self.total.as_array().T)

    def get(self) -> PairCountData:
        return PairCountData(
            binning=self.binning,
            count=self.count.as_array().sum(axis=0),
            total=self.total.as_array().sum(axis=0))

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, self.npatch, size=(n_boot, self.npatch))

    def get_jackknife_samples(
        self,
        global_norm: bool = False,
        **kwargs
    ) -> PairCountData:
        # The iterator expects a single patch index which is dropped in a single
        # realisation.
        count = []
        total = []
        if global_norm:
            global_total = self.total.as_array().sum(axis=0)
        for idx in range(self.npatch):  # leave-one-out iteration
            # we need to use the jackknife iterator twice
            patches = list(jackknife_iter(self.keys(), idx, mask=self.mask))
            count.append(self.count.sample(patches).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(patches).sum(axis=0))
        return PairCountData(
            binning=self.binning,
            count=np.array(count),
            total=np.array(total))

    def get_bootstrap_samples(
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
            count.append(self.count.sample(patches).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(patches).sum(axis=0))
        return PairCountData(
            binning=self.binning,
            count=np.array(count),
            total=np.array(total))

    def get_samples(
        self,
        method: str,
        **kwargs
    ) -> PairCountData:
        if method == "jackknife":
            samples = self.get_jackknife_samples(**kwargs)
        elif method == "bootstrap":
            samples = self.get_bootstrap_samples(**kwargs)
        else:
            raise NotImplementedError(
                f"sampling method '{method}' not implemented")
        return samples
