from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Interval, IntervalIndex
from treecorr import Catalog

from yet_another_wizz.utils import Timed


def _iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "right"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in zip(range(1, len(bins)), intervals):
        yield interval, bin_ids==i


class BinnedCatalog(Catalog):

    def __init__(
        self,
        file_name=None,
        config=None,
        *,
        num=0,
        logger=None,
        is_rand=False,
        x=None,
        y=None,
        z=None,
        ra=None,
        dec=None,
        r=None,
        w=None,
        wpos=None,
        flag=None,
        g1=None,
        g2=None,
        k=None,
        patch=None,
        patch_centers=None,
        rng=None,
        **kwargs
    ) -> None:
        with Timed("constructing patches"):
            super().__init__(
                file_name, config, num=num, logger=logger, is_rand=is_rand,
                x=x, y=y, z=z, ra=ra, dec=dec, r=r, w=w, wpos=wpos,
                flag=flag, g1=g1, g2=g2, k=k, patch=patch,
                patch_centers=patch_centers, rng=rng, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        patches: int | BinnedCatalog | str,
        ra: str,
        dec: str,
        redshift: str | None = None,
        w: str | None = None,
        **kwargs
    ) -> BinnedCatalog:
        if isinstance(patches, int):
            kwargs.update(dict(npatch=patches))
        elif isinstance(patches, str):
            kwargs.update(dict(patch=data[patches]))
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

    def __iter__(self) -> Iterator[BinnedCatalog]:
        for patch in self.get_patches(low_mem=True):
            yield patch

    def n_patches(self) -> int:
        return self.npatch

    @property
    def redshift(self) -> NDArray:
        return self.r

    def has_z(self) -> bool:
        return self.r is not None

    def get_z_min(self) -> float:
        try:
            return self.redshift.min()
        except AttributeError:
            return None

    def get_z_max(self) -> float:
        try:
            return self.redshift.max()
        except AttributeError:
            return None

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, Catalog]]:
        if self.redshift is None:
            raise ValueError("no redshifts for iteration provided")
        for interval, bin_mask in _iter_bin_masks(self.redshift, z_bins):
            new = self.copy()
            new.select(bin_mask)
            yield interval, BinnedCatalog.from_catalog(new)

    def patch_iter(self) -> Iterator[tuple[int, Catalog]]:
        patch_ids = sorted(set(self.patch))
        patches = self.get_patches()
        for patch_id, patch in zip(patch_ids, patches):
            yield patch_id, BinnedCatalog.from_catalog(patch)
