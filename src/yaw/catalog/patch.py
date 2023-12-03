from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator, overload

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


@dataclass
class PatchMetadata:
    length: int
    total: float | NotSet = field(default=NotSet)
    center: CoordSky | NotSet = field(default=NotSet)
    radius: DistSky | NotSet = field(default=NotSet)

    def compute_total(self, weight: NDArray | None) -> None:
        if weight is None:
            self.total = float(self.length)
        else:
            self.total = float(weight.sum())

    def compute_center_radius(
        self,
        ra: NDArray,
        dec: NDArray,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> None:
        coords = CoordSky(ra, dec)
        if center is None:
            self.center = coords.mean()
        else:
            self.center = center.to_sky()
        if radius is None:
            self.radius = coords.max_dist(self.center)
        else:
            self.radius = radius.to_sky()

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
        # replace center with ra/dec floats
        try:
            center: CoordSky = metadict.pop("center")
            metadict["ra"] = float(center.ra)
            metadict["dec"] = float(center.dec)
        except KeyError:
            pass
        # convert radius to float
        try:
            radius: DistSky = metadict.pop("radius")
            metadict["radius"] = float(radius.values)
        except KeyError:
            pass
        return metadict

    @classmethod
    def from_dict(cls, the_dict: dict) -> PatchMetadata:
        kwargs = {k: v for k, v in the_dict.items()}
        # reconstruct center
        try:
            kwargs["center"] = CoordSky(kwargs.pop("ra"), kwargs.pop("dec"))
        except KeyError:
            pass
        # reconstruct radius
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
            center = self.metadata.center
            if center is NotSet:
                center = None
            self.metadata.compute_center_radius(self.ra, self.dec, center)
            self._update_metadata_callback()
        return self.metadata.radius

    def iter_bins(
        self,
        z_bins: NDArray[np.float64],
        allow_no_redshift: bool = False,
    ) -> Generator[tuple[Interval, PatchData]]:
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
                - **cat** (:obj:`PatchCatalog`): instance containing the data
                  for this bin.
        """
        z_bins = np.asarray(z_bins)
        if not allow_no_redshift and not self.has_redshift:
            raise ValueError("no redshifts for iteration provdided")
        intervals = Binning.from_edges(z_bins, closed="left")
        if allow_no_redshift:
            for intv in intervals:
                yield intv, self
        else:
            redshift = self.redshift
            bin_index = intervals.apply(redshift)
            index_to_interval = dict(enumerate(intervals))
            data = utils.DataChunk(
                ra=self.ra,
                dec=self.dec,
                weight=self.weight,
                redshift=redshift,
                patch=bin_index,  # this is currently required by the implementation
            )
            for index, bin_data in data.groupby():
                if index < 0 or index >= len(intervals):
                    continue
                intv = index_to_interval[index]
                yield intv, PatchData(
                    self.id,
                    metadata=self.metadata.new_with_length(len(bin_data)),
                    **bin_data.to_dict(),
                )


class PatchDataResident(PatchData):
    pass


def load_attribute(
    path: TypePathStr, which: str, require: bool = True
) -> np.memmap | None:
    mempath = path / which
    try:
        return utils.memmap_load(mempath, np.float64)
    except FileNotFoundError as err:
        if require:
            raise FileNotFoundError(f"missing '{which}' data: {mempath}") from err


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
        self._init_data(metadata)

    @property
    def _path_metadata(self) -> Path:
        return self.path / "metadata.json"

    def _write_metadata(self) -> None:
        with open(self._path_metadata, "w") as f:
            the_dict = self.metadata.to_dict()
            json.dump(the_dict, f)

    def _update_metadata_callback(self) -> None:
        self._write_metadata()

    def _check_attr_size(
        self, attr: str, must_exist: bool, size: int | None = None
    ) -> int:
        try:
            fsize = os.path.getsize(self.path / attr) // 8  # is 8 byte double
        except FileNotFoundError:
            if must_exist:
                raise
        if size is None:
            return fsize
        elif fsize != size:
            raise ValueError(
                f"'{attr}' size ({fsize}) does not match expected size ({size})"
            )
        return size

    def _init_data(self, metadata: PatchMetadata | None) -> None:
        # check that ra and dec exist
        length = self._check_attr_size("ra", must_exist=True)
        self._check_attr_size("dec", must_exist=True, size=length)
        # check the optional data
        self._check_attr_size("weight", must_exist=False, size=length)
        self._check_attr_size("redshift", must_exist=False, size=length)
        # load the metadata
        if metadata is not None:
            self.metadata = metadata
            self._write_metadata()
        elif self._path_metadata.exists():
            with open(self._path_metadata) as f:
                the_dict = json.load(f)
            self.metadata = PatchMetadata.from_dict(the_dict)
        else:
            self.metadata = PatchMetadata(length)
            self._write_metadata()

    @classmethod
    def restore(cls, id: int, path: TypePathStr) -> PatchDataCached:
        new = cls.__new__(cls)
        new.path = Path(path)
        if not new.path.exists():
            raise FileNotFoundError(f"cache directory des not exist: {new.path}")
        new.id = id
        # populate the data attributes
        new._init_data(None)
        return new

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


class CacheWriter:
    def __init__(
        self,
        path: Path,
        has_weight: bool = False,
        has_redshift: bool = False,
        chunksize: int = 65_536,
    ) -> None:
        self.cache = {"ra": utils.ArrayCache(), "dec": utils.ArrayCache()}
        self.has_weight = has_weight
        if has_weight:
            self.cache["weight"] = utils.ArrayCache()
        self.has_redshift = has_redshift
        if has_redshift:
            self.cache["redshift"] = utils.ArrayCache()
        self.cachesize = 0
        self.chunksize = chunksize
        # create the cache directory
        self.path = Path(path)
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
        # check the data inputs
        chunk_dict = chunk.to_dict(drop_patch=True)
        has_weight = "weight" in chunk_dict
        if has_weight != self.has_weight:
            raise ValueError(
                f"writer has_weights={self.has_weight}, but chunk {has_weight=}"
            )
        has_redshift = "redshift" in chunk_dict
        if has_redshift != self.has_redshift:
            raise ValueError(
                f"writer has_weights={self.has_redshift}, but chunk {has_redshift=}"
            )
        # send data to cache
        self.cachesize += len(chunk)
        for key, cache in self.cache.items():
            cache.append(chunk_dict[key])
        # flush to disk
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


# the constructor functions


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
        for pid, patch_chunk in chunk.groupby():
            self._chunks[pid].append(patch_chunk)

    def close(self) -> None:
        pass

    def get_patches(self) -> dict[int, PatchDataResident]:
        data = {
            pid: DataChunk.from_chunks(chunks) for pid, chunks in self._chunks.items()
        }
        return {
            pid: PatchDataResident(pid, **data.to_dict()) for pid, data in data.items()
        }


class PatchWriter(Collector):
    def __init__(self, cache_directory: TypePathStr) -> None:
        self._writers: dict[int, CacheWriter] = dict()
        # set up cache directory
        self.cache_directory = Path(cache_directory)
        if not self.cache_directory.exists():
            self.cache_directory.mkdir(parents=True)

    def process(self, chunk: DataChunk) -> None:
        for pid, patch_chunk in chunk.groupby():
            if pid not in self._writers:
                cachepath = patch_path_from_id(self.cache_directory, pid)
                if os.path.exists(cachepath):
                    shutil.rmtree(cachepath)
                self._writers[pid] = CacheWriter(
                    cachepath,
                    has_weight=patch_chunk.weight is not None,
                    has_redshift=patch_chunk.redshift is not None,
                )
            self._writers[pid].append_chunk(patch_chunk)

    def get_patches(self) -> dict[int, PatchDataCached]:
        for writer in self._writers.values():
            writer.finalize()
        return {
            pid: PatchDataCached.restore(pid, writer.path)
            for pid, writer in self._writers.items()
        }
