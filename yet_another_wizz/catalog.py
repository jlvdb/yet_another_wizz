from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Interval


def iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "right"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = pd.IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in zip(range(1, len(bins)), intervals):
        yield interval, bin_ids==i


class BinnedCatalog:

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        patches: int | BinnedCatalog,
        ra: str,
        dec: str,
        redshift: str | None = None,
        w: str | None = None,
        **kwargs
    ) -> BinnedCatalog:
        if isinstance(patches, int):
            kwargs.update(dict(npatch=patches))
        else:
            kwargs.update(dict(patch_centers=patches.patch_centers))
        r = None if redshift is None else data[redshift]
        w = None if w is None else data[w]
        new = cls(
            ra=data[ra], ra_units="degrees",
            dec=data[dec], dec_units="degrees",
            r=r, w=w, **kwargs)
        return new

    @classmethod
    def from_catalog(cls, cat: Catalog) -> BinnedCatalog:
        new = cls.__new__(cls)
        new.__dict__ = cat.__dict__
        return new

    @property
    def redshift(self) -> NDArray:
        return self.r

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, Catalog]]:
        if self.redshift is None:
            raise ValueError("no redshifts for iteration provided")
        for interval, bin_mask in iter_bin_masks(self.redshift, z_bins):
            new = self.copy()
            new.select(bin_mask)
            yield interval, BinnedCatalog.from_catalog(new)

    def patch_iter(self) -> Iterator[tuple[int, Catalog]]:
        patch_ids = sorted(set(self.patch))
        patches = self.get_patches()
        for patch_id, patch in zip(patch_ids, patches):
            yield patch_id, BinnedCatalog.from_catalog(patch)