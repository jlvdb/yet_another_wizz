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
from yaw.catalog.kdtree import SphericalKDTree
from yaw.catalog.utils import DataChunk, patch_path_from_id
from yaw.config.default import NotSet
from yaw.core.containers import Binning, Interval
from yaw.core.coordinates import CoordSky, DistSky
from yaw.core.utils import TypePathStr

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from yaw.core.coordinates import Coordinate, Distance

__all__ = [
    "PatchMetadata",
    "PatchDataBuffered",
    "PatchDataCached",
    "patch_from_records",
    "patch_from_cache",
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


class PatchIpc:
    @abstractmethod
    def __init__(self, patch: PatchData) -> None:
        pass

    @abstractmethod
    def unpack(self) -> PatchData:
        pass


class PatchData(ABC):
    id: int
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    weight: NDArray[np.floating] | None
    redshift: NDArray[np.float64] | None
    metadata: PatchMetadata

    def __len__(self) -> int:
        return len(self.ra)

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
    ) -> Generator[tuple[Interval, PatchDataBuffered]]:
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
            bin_index = intervals.apply(self.redshift)
            index_to_interval = dict(enumerate(intervals))
            data = utils.DataChunk(
                ra=self.ra,
                dec=self.dec,
                weight=self.weight,
                redshift=self.redshift,
                patch=bin_index,  # this is currently required by the implementation
            )
            for index, bin_data in data.groupby():
                if index < 0 or index >= len(intervals):
                    continue
                intv = index_to_interval[index]
                metadata = copy(self.metadata)
                yield intv, PatchDataBuffered(
                    self.id, metadata=metadata, **bin_data.to_dict()
                )

    @abstractmethod
    def get_trees(
        self, z_bins: Binning | NDArray[np.float64] | None = None, **kwargs
    ) -> list[SphericalKDTree]:
        pass

    @abstractmethod
    def to_ipc(self) -> PatchIpc:
        pass


class PatchIpcBuffered:
    def __init__(self, patch: PatchDataBuffered) -> None:
        self.payload = patch

    def unpack(self) -> PatchDataBuffered:
        return self.payload


@dataclass(eq=False)
class PatchDataBuffered(PatchData):
    id: int
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    weight: NDArray[np.floating] | None = field(default=None)
    redshift: NDArray[np.float64] | None = field(default=None)
    metadata: PatchMetadata = field(default=None)

    def __post_init__(self) -> None:
        utils.check_arrays_matching_shape(self.ra, self.dec, self.weight, self.redshift)
        if self.metadata is None:
            self.metadata = PatchMetadata(len(self))

    def get_trees(
        self, z_bins: Binning | NDArray[np.float64] | None = None, **kwargs
    ) -> list[SphericalKDTree]:
        """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
        if z_bins is None:
            tree = SphericalKDTree(self.ra, self.dec, self.weight, **kwargs)
            tree._total = self.total  # no need to recompute this
            trees = [tree]
        else:
            trees = []
            for _, bindata in self.iter_bins(z_bins):
                tree = SphericalKDTree(
                    bindata.ra, bindata.dec, bindata.weight, **kwargs
                )
                tree._total = bindata.total  # will be recomputed for bin subset
                trees.append(tree)
        return trees

    def to_ipc(self) -> PatchIpcBuffered:
        return PatchIpcBuffered(self)


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


def load_attribute(
    path: TypePathStr, which: str, require: bool = True
) -> np.memmap | None:
    mempath = path / which
    try:
        return utils.memmap_load(mempath, np.float64)
    except FileNotFoundError as err:
        if require:
            raise FileNotFoundError(f"missing '{which}' data: {mempath}") from err


class PatchIpcCached:
    def __init__(self, patch: PatchDataCached) -> None:
        self.id = patch.id
        self.path = patch.path

    def unpack(self) -> PatchDataCached:
        return PatchDataCached.restore(self.id, self.path)


class PatchDataCached(PatchData):
    path: Path
    id: int
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    weight: NDArray[np.floating] | None
    redshift: NDArray[np.float64] | None
    metadata: PatchMetadata = field(default=None)
    _binning: Binning | None | NotSet = field(default=NotSet, init=False)

    def __init__(
        self,
        path: TypePathStr,
        id: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.floating] | None = None,
        redshift: NDArray[np.float64] | None = None,
        metadata: PatchMetadata | None = None,
    ) -> PatchDataCached:
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

    def _init_data(self, metadata: PatchMetadata | None) -> None:
        # check that ra and dec exist and load them
        self.ra = load_attribute(self.path, "ra", require=True)
        self.dec = load_attribute(self.path, "dec", require=True)
        # add the optional data, otherwise initialised to None
        self.weight = load_attribute(self.path, "weight", require=False)
        self.redshift = load_attribute(self.path, "redshift", require=False)
        # run final checks and load metadata
        utils.check_arrays_matching_shape(self.ra, self.dec, self.weight, self.redshift)
        if metadata is not None:
            self.metadata = metadata
            self._write_metadata()
        elif self._path_metadata.exists():
            with open(self._path_metadata) as f:
                the_dict = json.load(f)
            self.metadata = PatchMetadata.from_dict(the_dict)
        else:
            self.metadata = PatchMetadata(len(self))
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
    def _path_metadata(self) -> Path:
        return self.path / "metadata.json"

    def _write_metadata(self) -> None:
        with open(self._path_metadata, "w") as f:
            the_dict = self.metadata.to_dict()
            json.dump(the_dict, f)

    def _update_metadata_callback(self) -> None:
        self._write_metadata()

    @property
    def _path_binning(self) -> Path:
        return self.path / "binning.pickle"

    def _get_binning(self) -> Binning | None:
        if self._binning is NotSet:
            self._binning = utils.read_pickle(self._path_binning)
        return self._binning

    def _set_binning(self, z_bins: Binning | NDArray[np.float64] | None) -> None:
        if z_bins is not None and not isinstance(z_bins, Binning):
            z_bins = Binning.from_edges(z_bins)
        utils.write_pickle(self._path_binning, z_bins)
        self._binning = z_bins

    @property
    def _path_trees(self) -> Path:
        return self.path / "trees.pickle"

    def _read_trees(self) -> list[SphericalKDTree]:
        return utils.read_pickle(self._path_trees)

    def _write_trees(self, trees: list[SphericalKDTree]) -> None:
        utils.write_pickle(self._path_trees, trees)

    def _trees_cached(self, z_bins: Binning | NDArray[np.float64] | None) -> bool:
        # check if any data is cached
        if not self._path_binning.exists() or not self._path_trees.exists():
            return False
        # compare the binning
        binning = self._get_binning()
        if binning is None:
            binning_equal = z_bins is None
        elif z_bins is None:
            binning_equal = False
        elif isinstance(z_bins, Binning):
            binning_equal = (
                (z_bins.closed == binning.closed)
                & np.any(z_bins.left == binning.left)
                & np.any(z_bins.right == binning.right)
            )
        else:
            binning_equal = binning.edges_equal(z_bins)
        return binning_equal

    def get_trees(
        self, z_bins: Binning | NDArray[np.float64] | None = None, **kwargs
    ) -> list[SphericalKDTree]:
        if self._trees_cached(z_bins):
            trees = self._read_trees()
        else:
            trees = super().get_trees(z_bins=z_bins, **kwargs)
            self._write_trees(trees)
            self._set_binning(z_bins)
        return trees

    def to_ipc(self) -> PatchIpcCached:
        return PatchIpcCached(self)


# the constructor functions


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
) -> PatchDataBuffered:
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
) -> PatchDataBuffered:
    ...


def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr | None = None,
) -> PatchDataBuffered | PatchDataCached:
    if cachepath is None:
        return PatchDataBuffered(
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


def patch_from_cache(id: int, cachepath: TypePathStr) -> PatchDataCached:
    return PatchDataCached.restore(id, cachepath)


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

    def get_patches(self) -> dict[int, PatchDataBuffered]:
        data = {
            pid: DataChunk.from_chunks(chunks) for pid, chunks in self._chunks.items()
        }
        return {
            pid: PatchDataBuffered(pid, **data.to_dict()) for pid, data in data.items()
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
