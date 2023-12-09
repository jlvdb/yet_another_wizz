from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator, TypeVar, overload

import numpy as np

from yaw.catalog import utils
from yaw.catalog.utils import DataChunk, patch_path_from_id
from yaw.config.default import NotSet
from yaw.core.containers import Binning, Interval
from yaw.core.coordinates import CoordSky, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.core.coordinates import Coordinate, Distance
    from yaw.core.utils import TypePathStr

__all__ = [
    "PatchMetadata",
    "PatchDataResident",
    "PatchDataCached",
    "patch_from_records",
    "PatchCollector",
    "PatchWriter",
]


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


class PatchDataResident(PatchData):
    pass  # alias to express data layout through type instead of state


def get_and_check_data_size(
    path: TypePathStr,
    *,
    must_exist: bool,
    require_size: int | None = None,
    itemsize: int = 8,
) -> int | None:
    if path.exists():
        nbytes = os.path.getsize(path)
        size = nbytes // itemsize
        if size != require_size:
            raise ValueError("file size does not match expected size")
        return size

    elif must_exist:
        raise FileNotFoundError(str(path))


class PatchDataCached(PatchData):
    def __init__(
        self,
        path: TypePathStr,
        id: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.floating] | None = None,
        redshift: NDArray[np.float64] | None = None,
        metadata: PatchMetadata | None = None,
    ) -> None:
        self.id = id
        has_weight = weight is not None
        has_redshift = redshift is not None
        # create memory maps for the input data
        writer = CacheWriter(path, has_weight, has_redshift)
        writer.append_data(ra, dec, weight, redshift)
        writer.finalize()
        self.path = writer.path
        # populate the data attributes
        self._init_from_disk(metadata)

    @classmethod
    def restore(cls, id: int, path: TypePathStr) -> PatchDataCached:
        new = cls.__new__(cls)
        new.path = Path(path)
        if not new.path.exists():
            raise FileNotFoundError(f"cache directory des not exist: {new.path}")
        new.id = id
        # populate the data attributes
        new._init_from_disk(metadata=None)
        return new

    @property
    def _path_metadata(self) -> Path:
        return self.path / "metadata.json"

    def _write_metadata(self) -> None:
        the_dict = self.metadata.to_dict()
        with open(self._path_metadata, "w") as f:
            json.dump(the_dict, f)

    _update_metadata_callback = _write_metadata

    def _check_data_files(self) -> None:
        # required data
        length = get_and_check_data_size("ra", must_exist=True)
        get_and_check_data_size("dec", must_exist=True, require_size=length)
        # optional data
        get_and_check_data_size("weight", must_exist=False, require_size=length)
        get_and_check_data_size("redshift", must_exist=False, require_size=length)
        return length

    def _init_metadata(self, data_length: int, metadata: PatchMetadata | None) -> None:
        overwrite = metadata is not None
        load_existing = self._path_metadata.exists()

        if overwrite:
            self.metadata = metadata
            self._write_metadata()

        elif load_existing:
            with open(self._path_metadata) as f:
                the_dict = json.load(f)
            self.metadata = PatchMetadata.from_dict(the_dict)

        else:
            self.metadata = PatchMetadata(data_length)
            self._write_metadata()

    def _init_from_disk(self, metadata: PatchMetadata | None) -> None:
        length = self._check_data_files()
        self._init_metadata(length, metadata)

    @property
    def ra(self) -> NDArray[np.float64]:
        np.fromfile(self.path / "ra", dtype=np.float64)

    @property
    def dec(self) -> NDArray[np.float64]:
        np.fromfile(self.path / "dec", dtype=np.float64)

    @property
    def weight(self) -> NDArray[np.float64] | None:
        try:
            np.fromfile(self.path / "weight", dtype=np.float64)
        except FileNotFoundError:
            return None

    @property
    def redshift(self) -> NDArray[np.float64] | None:
        try:
            np.fromfile(self.path / "redshift", dtype=np.float64)
        except FileNotFoundError:
            return None


@dataclass
class CacheWriter:
    path: TypePathStr
    has_weight: bool = field(default=False)
    has_redshift: bool = field(default=False)
    chunksize: int = field(default=65_536)

    def __post_init__(self) -> None:
        self.cachesize = 0
        self.cache = {"ra": utils.ArrayCache(), "dec": utils.ArrayCache()}
        if self.has_weight:
            self.cache["weight"] = utils.ArrayCache()
        if self.has_redshift:
            self.cache["redshift"] = utils.ArrayCache()

        self.path = Path(self.path)
        if self.path.exists():
            raise FileExistsError(f"directory already exists: {self.path}")
        self.path.mkdir(parents=True)

    def flush(self):
        for key, cache in self.cache.items():
            with open(self.path / key, mode="a") as f:
                cache.get_values().tofile(f)
            cache.clear()
        self.cachesize = 0

    def append_chunk(self, chunk: DataChunk) -> None:
        chunk_dict = chunk.to_dict(drop_patch=True)
        utils.check_optional_args(chunk_dict["weight"], self.has_weight, "weight")
        utils.check_optional_args(chunk_dict["redshift"], self.has_redshift, "redshift")

        self.cachesize += len(chunk)
        for key, cache in self.cache.items():
            cache.append(chunk_dict[key])
        if self.cachesize > self.chunksize:
            self.flush()

    def append_data(
        self,
        ra: NDArray,
        dec: NDArray,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
    ) -> None:
        self.append_chunk(DataChunk(ra=ra, dec=dec, weight=weight, redshift=redshift))

    def finalize(self) -> None:
        self.flush()


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


class PatchCollector(Collector):
    def __init__(self) -> None:
        self._chunks: dict[int, list[DataChunk]] = defaultdict(list)

    def process(self, chunk: DataChunk) -> None:
        for patch_id, patch_chunk in chunk.groupby():
            self._chunks[patch_id].append(patch_chunk)

    def close(self) -> None:
        pass

    def get_patches(self) -> dict[int, PatchDataResident]:
        concatenated = {
            patch_id: DataChunk.from_chunks(chunks)
            for patch_id, chunks in self._chunks.items()
        }
        return {
            patch_id: PatchDataResident(patch_id, **chunk.to_dict())
            for patch_id, chunk in concatenated.items()
        }


class PatchWriter(Collector):
    def __init__(self, cache_directory: TypePathStr) -> None:
        self._writers: dict[int, CacheWriter] = dict()

        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def _get_and_delete_cache_path(self, patch_id: int) -> Path:
        cachepath = patch_path_from_id(self.cache_directory, patch_id)
        if os.path.exists(cachepath):
            shutil.rmtree(cachepath)

    def process(self, chunk: DataChunk) -> None:
        for patch_id, patch_chunk in chunk.groupby():
            if patch_id not in self._writers:
                cachepath = self._get_and_delete_cache_path(patch_id)
                self._writers[patch_id] = CacheWriter(
                    cachepath,
                    has_weight=patch_chunk.weight is not None,
                    has_redshift=patch_chunk.redshift is not None,
                )
            self._writers[patch_id].append_chunk(patch_chunk)

    def get_patches(self) -> dict[int, PatchDataCached]:
        for writer in self._writers.values():
            writer.finalize()
        return {
            patch_id: PatchDataCached.restore(patch_id, writer.path)
            for patch_id, writer in self._writers.items()
        }


# the Patch constructor functions


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
) -> PatchDataResident:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr = ...,
) -> PatchDataCached:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: None = None,
) -> PatchDataResident:
    ...


def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr | None = None,
) -> PatchDataResident | PatchDataCached:
    if cachepath is None:
        return PatchDataResident(
            id=id, ra=ra, dec=dec, weight=weight, redshift=redshift, metadata=metadata
        )
    else:
        return PatchDataCached(
            cachepath,
            id=id,
            ra=ra,
            dec=dec,
            weight=weight,
            redshift=redshift,
            metadata=metadata,
        )
