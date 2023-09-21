from __future__ import annotations

import gc
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from yaw.core.coordinates import Coordinate, CoordSky, Distance, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval

__all__ = ["PatchBase"]


class CachingError(Exception):
    pass


class PatchBase(ABC):
    """TODO"""

    id: int = 0
    """Unique index of the patch."""
    cachefile: str = None
    """The path to the cached .feather data file if caching is enabled."""
    _has_z: bool = False
    _has_weights: bool = False
    _len: int = 0
    _total: float = None
    _center: CoordSky = None
    _radius: DistSky = None
    _data: Any = None

    def __init__(
        self,
        id: int,
        ra: NDArray[np.float_],
        dec: NDArray[np.float_],
        redshifts: NDArray[np.float_] | None = None,
        weights: NDArray[np.float_] | None = None,
        cachefile: str | None = None,
        center: Coordinate | None = None,
        radius: Distance | None = None,
        degrees: bool = True,
    ) -> None:
        """Create a new patch from a data frame.

        Coordiantes are converted to radian. If a cache path is provided, a
        cache file is created and the data is dropped from memory.

        Args:
            id (:obj:`int`):
                Unique index of the patch.
            data (:obj:`pandas.DataFrame`):
                Data frame with columns ``ra``, ``dec`` (by default assumed to
                be in degrees) and optionally ``weights``, ``redshift`` if
                either data is available.
            cachefile (:obj:`str`, optional):
                If provided, the data is cached as .feather file at this path.
            center (:obj:`yaw.core.coordiante.Coordiante`, optional):
                Center coordinates of the patch. Computed automatically if not
                provided.
            radius (:obj:`yaw.core.coordiante.Distance`, optional):
                The angular size of the patch. Computed automatically if not
                provided.
            degrees (:obj:`bool`):
                Whether the input coordinates ``ra``, ``dec`` are in degrees.
        """
        self.id = id
        # collect the data
        if degrees:
            ra = np.deg2rad(ra)
            dec = np.deg2rad(dec)
        data = dict(
            ra=ra, dec=dec, redshifts=np.asarray(redshifts), weights=np.asarray(weights)
        )

        # compute the metadata
        self._has_z = redshifts is not None
        self._has_weights = weights is not None
        self._len = len(ra)
        self._total = float(weights.sum() if self._has_weights else self._len)

        # precompute (estimate) the patch center and size since it is quite fast
        # and the data is still loaded
        if center is None or radius is None:
            SUBSET_SIZE = 1000  # seems a reasonable, fast but not too sparse
            if self._len < SUBSET_SIZE:
                positions = CoordSky(ra, dec).to_3d()
            else:
                rng = np.random.default_rng(seed=12345)
                which = rng.integers(0, self._len, size=SUBSET_SIZE)
                positions = CoordSky(ra[which], dec[which]).to_3d()

        # store center in xyz coordinates
        if center is None:
            self._center = positions.mean()
        else:
            self._center = center.to_3d()

        if center is None or radius is None:  # new center requires recomputing
            # compute maximum distance to any of the data points
            radius = positions.distance(self._center).max()
        # store radius in radians
        self._radius = radius.to_sky()

        self._init(data, cachefile)

    @abstractmethod
    def _init(
        self, data: dict[str, NDArray[np.float_] | None], cachefile: str | None = None
    ) -> None:
        pass

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, loaded={self.is_loaded()})"
        return s

    def __len__(self) -> int:
        return self._len

    @abstractclassmethod
    def from_cached(
        cls,
        cachefile: str,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> PatchBase:
        """Restore the patch instance from its cache file.

        Optionally, the center and radius of the patch can be provided to avoid
        recomputing these quantities.

        Args:
            cachefile (:obj:`str`):
                Path to the cach file (.feather)
            center (:obj:`yaw.core.coordiante.Coordiante`, optional):
                Center coordinates of the patch. Computed automatically if not
                provided.
            radius (:obj:`yaw.core.coordiante.Distance`, optional):
                The angular size of the patch. Computed automatically if not
                provided.
        """
        pass

    def is_loaded(self) -> bool:
        """Whether the data is present in memory"""
        return self._data is not None

    def require_loaded(self) -> None:
        """Raise a :obj:`CachingError` if the data is not present in memory."""
        if not self.is_loaded():
            raise CachingError("data is not loaded")

    @abstractmethod
    def load(self, use_threads: bool = True) -> None:
        """Load the data from the cache file into memory.

        Raises a :obj:`CachingError` if no cache file is sepcified.
        """
        if not self.is_loaded():
            if self.cachefile is None:
                raise CachingError("no datapath provided to load the data")
            self._data = pd.read_feather(self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        """Drop the data from memory.

        Raises a :obj:`CachingError` if no cache file is sepcified.
        """
        if self.cachefile is None:
            raise CachingError("no datapath provided to unload the data")
        self._data = None
        gc.collect()

    def has_redshifts(self) -> bool:
        """Whether the patch data include redshifts."""
        return self._has_z

    def has_weights(self) -> bool:
        """Whether the patch data include weights."""
        return self._has_weights

    @property
    def data(self) -> DataFrame:
        """Direct access to the underlying :obj:`pandas.DataFrame` which holds
        the patch data."""
        self.require_loaded()
        columns = dict(ra=self.ra, dec=self.dec)
        if self.has_redshifts():
            columns["redshifts"] = self.redshifts
        if self.has_weights():
            columns["weights"] = self.weights
        return DataFrame(**columns)

    @abstractproperty
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians.

        Raises a :obj:`CachingError` if data is not loaded.
        """
        self.require_loaded()

    @abstractproperty
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians.

        Raises a :obj:`CachingError` if data is not loaded.
        """
        self.require_loaded()

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Raises a :obj:`CachingError` if data is not loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky(self.ra, self.dec)

    @abstractproperty
    def redshifts(self) -> NDArray[np.float_]:
        """Get the redshifts as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded.
        """
        pass

    @abstractproperty
    def weights(self) -> NDArray[np.float_]:
        """Get the object weights as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded.
        """
        pass

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available.

        Available even if no data is loaded.
        """
        return self._total

    @property
    def center(self) -> CoordSky:
        """Get the patch centers in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return self._center.to_sky()

    @property
    def radius(self) -> DistSky:
        """Get the patch size in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return self._radius

    def iter_bins(
        self, z_bins: NDArray[np.float_], allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, PatchBase]]:
        """Iterate the patch in bins of redshift.

        Args:
            z_bins (:obj:`NDArray`):
                Edges of the redshift bins.
            allow_no_redshift (:obj:`bool`):
                If true and the data has no redshifts, the iterator yields the
                whole patch at each iteration step.

        Yields:
            (tuple): tuple containing:
                - **intv** (:obj:`pandas.Interval`): the selection for this bin.
                - **cat** (:obj:`PatchBase`): instance containing the data
                  for this bin.
        """
        if not allow_no_redshift and not self.has_redshifts():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in pd.IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for intv, bin_data in self.data.groupby(pd.cut(self.redshifts, z_bins)):
                datacols = {col: bin_data[col] for col in bin_data.columns}
                yield intv, self.__class__(
                    self.id,
                    **datacols,
                    cachefile=None,
                    degrees=False,
                    center=self._center,
                    radius=self._radius,
                )
