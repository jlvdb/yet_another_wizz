from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import treecorr
from pandas import Interval, IntervalIndex

from yet_another_wizz.resampling import ArrayDict
from yet_another_wizz.resampling import PairCountResult


class PairCountResultTC(PairCountResult):

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: treecorr.NNCorrelation
    ) -> PairCountResult:
        # extract the (cross-patch) pair counts
        npatch = max(correlation.npatch1, correlation.npatch2)
        keys = []
        count = np.empty((len(correlation.results), 1))
        total = np.empty((len(correlation.results), 1))
        for i, (patches, result) in enumerate(correlation.results.items()):
            keys.append(patches)
            count[i] = result.weight
            total[i] = result.tot
        return cls(
            npatch=npatch,
            count=ArrayDict(keys, count),
            total=ArrayDict(keys, total),
            mask=correlation._ok,
            binning=IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[PairCountResult]
    ) -> PairCountResult:
        # check that the data is compatible
        if len(zbins) == 0:
            raise ValueError("'zbins' is empty")
        npatch = zbins[0].npatch
        mask = zbins[0].mask
        keys = tuple(zbins[0].keys())
        nbins = zbins[0].nbins
        for zbin in zbins[1:]:
            if zbin.npatch != npatch:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
            if tuple(zbin.keys()) != keys:
                raise ValueError("patches are inconsistent")
            if zbin.nbins != nbins:
                raise IndexError("number of bins is inconsistent")

        # check the ordering of the bins based on the provided intervals
        binning = IntervalIndex.from_tuples([
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
            npatch=npatch,
            count=count,
            total=total,
            mask=mask,
            binning=binning)
