from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Generator, TypeVar, overload

import numpy as np

from yaw.catalog import utils
from yaw.config.default import NotSet
from yaw.core.containers import Binning, Interval
from yaw.core.coordinates import CoordSky, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.utils import DataChunk
    from yaw.core.coordinates import Coordinate, Distance


T = TypeVar("T")


@overload
def ensure_notset_is_none(obj: None) -> None:
    ...


@overload
def ensure_notset_is_none(obj: NotSet) -> None:
    ...


@overload
def ensure_notset_is_none(obj: T) -> T:
    ...


@overload
def ensure_notset_is_none(obj: T | NotSet | None) -> T | None:
    return None if obj is None or obj is NotSet else obj


class Collector(ABC):
    @abstractmethod
    def process(
        self,
        chunk: DataChunk,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def get_patches(self) -> dict[int, PatchData]:
        pass


@dataclass(frozen=True)
class PatchMetadata:
    length: int
    total: float | NotSet = field(default=NotSet)
    center: CoordSky | NotSet = field(default=NotSet)
    radius: DistSky | NotSet = field(default=NotSet)

    def set_center(self, center: Coordinate):
        object.__setattr__(self, "center", center.to_sky())

    def compute_total(self, weight: NDArray | None) -> None:
        if weight is None:
            total = float(self.length)
        else:
            total = float(weight.sum())
        object.__setattr__(self, "total", total)

    def compute_center_radius(
        self,
        ra: NDArray,
        dec: NDArray,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> None:
        coords = CoordSky(ra, dec)
        center = coords.mean() if center is None else center.to_sky()
        radius = coords.max_dist(center) if radius is None else radius.to_sky()
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    @classmethod
    def new_with_length(self, length: int) -> PatchMetadata:
        return PatchMetadata(
            length=length,
            total=NotSet,
            center=copy(self.center),
            radius=copy(self.radius),
        )

    def to_dict(self) -> dict:
        metadict = {
            attr: value for attr, value in asdict(self).items() if value is not NotSet
        }

        try:
            center: CoordSky = metadict.pop("center")
            metadict["ra"] = float(center.ra)
            metadict["dec"] = float(center.dec)
        except KeyError:
            pass

        try:
            radius: DistSky = metadict.pop("radius")
            metadict["radius"] = float(radius.values)
        except KeyError:
            pass

        return metadict

    @classmethod
    def from_dict(cls, the_dict: dict) -> PatchMetadata:
        kwargs = {k: v for k, v in the_dict.items()}

        try:
            kwargs["center"] = CoordSky(kwargs.pop("ra"), kwargs.pop("dec"))
        except KeyError:
            pass

        try:
            kwargs["radius"] = DistSky(kwargs["radius"])
        except KeyError:
            pass

        return cls(**kwargs)


class PatchData:
    def __init__(
        self,
        id: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.floating] | None = None,
        redshift: NDArray[np.float64] | None = None,
        metadata: PatchMetadata | None = None,
    ) -> None:
        self.id = id
        # set data fields
        utils.check_arrays_matching_shape(ra, dec, weight, redshift)
        self.ra = ra
        self.dec = dec
        self.weight = weight
        self.redshift = redshift
        # set metadata
        if metadata is None:
            metadata = PatchMetadata(len(ra))
        self.metadata = metadata

    def __len__(self) -> int:
        return self.metadata.length

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, redshifts={self.has_redshift})"
        return s

    @property
    def has_weight(self) -> bool:
        return self.weight is not None

    @property
    def has_redshift(self) -> bool:
        return self.redshift is not None

    def _update_metadata_callback(self) -> None:
        pass

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available.

        Available even if no data is loaded."""
        if self.metadata.total is NotSet:
            self.metadata.compute_total(self.weight)
            self._update_metadata_callback()
        return self.metadata.total

    @property
    def center(self) -> CoordSky:
        """Get the patch centers in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        if self.metadata.center is NotSet:
            self.metadata.compute_center_radius(self.ra, self.dec)
            self._update_metadata_callback()
        return self.metadata.center

    @property
    def radius(self) -> DistSky:
        """Get the patch size in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        if self.metadata.radius is NotSet:
            center = ensure_notset_is_none(self.metadata.center)
            self.metadata.compute_center_radius(self.ra, self.dec, center)
            self._update_metadata_callback()
        return self.metadata.radius

    def iter_bins(
        self,
        z_bins: NDArray[np.float64],
        require_redshift: bool = True,
    ) -> Generator[tuple[Interval, PatchData]]:
        """Iterate the patch in bins of redshift.

        Args:
            z_bins (:obj:`NDArray`):
                Edges of the redshift bins.
            require_redshift (:obj:`bool`):
                If false and the data has no redshifts, the iterator yields the
                whole patch at each iteration step.

        Yields:
            (tuple): tuple containing:
                - **intv** (:obj:`pandas.Interval`): the selection for this bin.
                - **cat** (:obj:`PatchCatalog`): instance containing the data
                  for this bin.
        """
        if require_redshift and not self.has_redshift:
            raise ValueError("no redshifts for iteration provdided")
        intervals = Binning.from_edges(np.asarray(z_bins), closed="left")
        n_bins = len(intervals)

        if require_redshift:
            bin_index = intervals.apply(self.redshift)
            index_to_interval = dict(enumerate(intervals))
            data = utils.DataChunk(
                ra=self.ra,
                dec=self.dec,
                weight=self.weight,
                redshift=self.redshift,
                patch=bin_index,  # this is currently required by .groupby()
            )
            for index, bin_data in data.groupby():
                if 0 <= index < n_bins:
                    metadata = self.metadata.new_with_length(len(bin_data))
                    kwargs = bin_data.to_dict()
                    patch_bin = PatchData(self.id, metadata=metadata, **kwargs)
                    yield index_to_interval[index], patch_bin

        else:
            for intv in intervals:
                yield intv, self
