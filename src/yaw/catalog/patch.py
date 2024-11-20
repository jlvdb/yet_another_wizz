"""
Implements a patch of catalog data used in correlation measurements for spatial
resampling.

Each data catalog consists of a number of patches. Each patch stores a portion
of the catalog data, which is cached on disk. Patch data is never permanently
held im memory, but loaded from a patch's cache directory on request.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from yaw.coordinates import AngularCoordinates, AngularDistances
from yaw.datachunk import DataChunk, DataChunkInfo, HandlesDataChunk
from yaw.utils.abc import YamlSerialisable

if TYPE_CHECKING:
    from io import TextIOBase
    from typing import Any

    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.datachunk import TypeDataChunk

__all__ = [
    "Metadata",
    "Patch",
    "PatchWriter",
]

PATCH_DATA_FILE = "data.bin"
"""File name in cache of patch which contains the catalog data."""

logger = logging.getLogger(__name__)


class Metadata(YamlSerialisable):
    """
    Container for patch meta data.

    Bundles the number of records stored in the patch, the sum of weights, and
    distribution of objects on sky through the center point and containing
    radius.

    Args:
        num_records:
            Number of data points in the patch.
        sum_weights:
            Sum of point weights or same as :obj:`num_records`.
        center:
            Center point (mean) of all data points,
            :obj:`~yaw.AngularCoordinates` in radian.
        radius:
            Radius around center point containing all data points,
            :obj:`~yaw.AngularDistances` in radian.
    """

    __slots__ = (
        "num_records",
        "sum_weights",
        "center",
        "radius",
    )

    num_records: int
    """Number of data points in the patch."""
    sum_weights: float
    """Sum of point weights."""
    center: AngularCoordinates
    """Center point (mean) of all data points."""
    radius: AngularDistances
    """Radius around center point containing all data points."""

    def __init__(
        self,
        *,
        num_records: int,
        sum_weights: float,
        center: AngularCoordinates,
        radius: AngularDistances,
    ) -> None:
        self.num_records = num_records
        self.sum_weights = sum_weights
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"sum_weights={self.sum_weights}",
            f"center={self.center.data[0]}",
            f"radius={self.radius.data[0]}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @classmethod
    def compute(
        cls,
        coords: AngularCoordinates,
        *,
        weights: NDArray | None = None,
        center: AngularCoordinates | None = None,
    ) -> Metadata:
        """
        Compute the meta data from the patch data.

        If no weights are provided, the sum of weights will equal the number of
        data points. Weights are also used when computing the center point.

        Args:
            coords:
                Coordinates of patch data points, given as
                :obj:`~yaw.AngularCoordinates`.

        Keyword Args:
            weights:
                Optional, weights of data points.
            center:
                Optional, use this specific center point, e.g. when using an
                externally computed patch center.

        Returns:
            Final instance of meta data.
        """
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.sum_weights = float(new.num_records)
        else:
            new.sum_weights = float(np.sum(weights))

        if center is not None:
            if len(center) != 1:
                raise ValueError("'center' must be one single coordinate")
            new.center = center.copy()
        else:
            new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()

        return new

    @classmethod
    def from_dict(cls, kwarg_dict: dict) -> Metadata:
        center = AngularCoordinates(kwarg_dict.pop("center"))
        radius = AngularDistances(kwarg_dict.pop("radius"))
        return cls(center=center, radius=radius, **kwarg_dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            num_records=int(self.num_records),
            sum_weights=float(self.sum_weights),
            center=self.center.tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


def read_patch_data(path: Path | str) -> tuple[DataChunkInfo, TypeDataChunk]:
    """
    Read the binary catalog data stored in cache of a patch.

    Returns:
        A numpy array with composite data type. The fields represent the
        different data columns and can be ``ra``, ``dec``, ``weights``, and
        ``redshifts``, where the latter two are optional.
    """
    with open(path, mode="rb") as f:
        data_attrs = DataChunkInfo.from_bytes(f.read(1))
        dtype = np.dtype([(attr, "f8") for attr in data_attrs.get_list()])
        rawdata = np.fromfile(f, dtype=np.byte)

    return data_attrs, rawdata.view(dtype)


class PatchWriter(AbstractContextManager, HandlesDataChunk):
    """
    Incrementally writes catalog data for a single patch.

    Data is received in chunks and cached internally. When the cache is full or
    the writer is closed, data is written as plain binary data to disk.

    Args:
        cache_path:
            Path to directory that serves as cache, must not exist.

    Keyword Args:
        chunk_info:
            An instance of :obj:`yaw.datachunk.DataChunkInfo` indicating which
            optional data attributes are processed by the pipeline.
        buffersize:
            Optional, maximum number of records to store in the internal cache.

    Raises:
        FileExistsError:
            If the cache directory already exists.

    Attributes:
        cache_path:
            The cache directory into which catalog data written.
        buffersize:
            The maximum number of records in the internal cache, which gets
            flushed if the cache size is exceeded.
    """

    __slots__ = (
        "cache_path",
        "buffersize",
        "_num_processed",
        "_shards",
        "_file",
    )

    def __init__(
        self,
        cache_path: Path | str,
        *,
        chunk_info: DataChunkInfo,
        buffersize: int = 65_536,
    ) -> None:
        self.buffersize = int(buffersize)
        self._shards = []
        self._num_processed = 0

        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)

        self._file = None
        self.open()
        header = chunk_info.to_bytes()
        self._file.write(header)
        self._chunk_info = chunk_info

    def __repr__(self) -> str:
        items = (
            f"buffersize={self.buffersize}",
            f"cachesize={self.cachesize}",
            f"processed={self._num_processed}",
        )
        attrs = self._chunk_info.format()
        return f"{type(self).__name__}({', '.join(items)}, {attrs}) @ {self.cache_path}"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    @property
    def data_path(self) -> Path:
        """Path to binary file with patch data."""
        return self.cache_path / PATCH_DATA_FILE

    @property
    def cachesize(self) -> int:
        """The current number of records stored in the internal cache."""
        return sum(len(shard) for shard in self._shards)

    @property
    def num_processed(self) -> int:
        """The current number of records written to disk."""
        return self._num_processed

    def open(self) -> TextIOBase:
        """If it did not already happened, opens the target file for writing."""
        if self._file is None:
            self._file = self.data_path.open(mode="ab")
        return self._file

    def flush(self) -> None:
        """Flush internal cache to disk."""
        if len(self._shards) > 0:
            data = np.concatenate(self._shards)
            self._num_processed += len(data)
            self._shards = []

            data.tofile(self.open())  # ensure file is ready for writing

    def close(self) -> None:
        """Flushes the internal cache and closes the file."""
        self.flush()
        self._file.close()
        self._file = None

    def process_chunk(self, data: TypeDataChunk) -> None:
        """
        Process a new chunk of catalog data.

        The data must be provided as a single numpy array with columns ``ra``,
        ``dec`` and optionally also ``weights`` and ``redshifts``. Data is
        cached internally or flushed to disk if the maximum cache size is
        exceeded.

        Args:
            data:
                Numpy array with data records as described above.
        """
        if DataChunk.hasattr(data, "patch_ids"):
            raise ValueError("'patch_ids' field must stripped before writing data")

        self._shards.append(data)

        if self.cachesize >= self.buffersize:
            self.flush()


class Patch(HandlesDataChunk):
    """
    A single spatial patch of catalog data.

    Data has point coordinates and optionally weights and redshifts. This data
    is cached on disk in a binary file (``data.bin``) that is read when
    accessing any of the classes data attributes. Additionaly meta data, such as
    the patch center and radius, that describe the spatial distribution of the
    contained data points, are availble and stored as YAML file (``meta.yml``)

    The cached data is organised in a single directory as follows::

        [cache_path]/
          ├╴ data.bin
          ├╴ meta.yml
          ├╴ binning   (created by trees.BinnedTrees)
          └╴ trees.pkl (created by trees.BinnedTrees)

    Supports efficient pickeling as long as the cached data is not deleted or
    moved.
    """

    __slots__ = ("meta", "cache_path", "_chunk_info")

    meta: Metadata
    """Patch meta data; number of records stored in the patch, the sum of
    weights, and distribution of objects on sky through the center point and
    containing radius."""

    cache_path: Path
    """Directory where (meta) data is cached."""

    def __init__(
        self, cache_path: Path | str, center: AngularCoordinates | None = None
    ) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)
            with self.data_path.open(mode="rb") as f:
                self._chunk_info = DataChunkInfo.from_bytes(f.read(1))

        except FileNotFoundError:
            self._chunk_info, data = read_patch_data(self.data_path)
            self.meta = Metadata.compute(
                DataChunk.get_coords(data),
                weights=DataChunk.getattr(data, "weights", None),
                center=center,
            )
            self.meta.to_file(meta_data_file)

    def __repr__(self) -> str:
        num = f"num_records={self.meta.num_records}"
        attrs = self._chunk_info.format()
        return f"{type(self).__name__}({num}, {attrs}) @ {self.cache_path}"

    def __getstate__(self) -> dict:
        return dict(
            cache_path=self.cache_path,
            meta=self.meta,
            _chunk_info=self._chunk_info,
        )

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    @property
    def data_path(self) -> Path:
        """Path to binary file with patch data."""
        return self.cache_path / PATCH_DATA_FILE

    @property
    def has_patch_ids(self) -> Literal[False]:
        """Patches never provide patch IDs."""
        return False

    def load_data(self) -> TypeDataChunk:
        """
        Load the cached object data with coordinates and optional weights and
        redshifts.

        Returns:
            A numpy array with composite data type. The fields represent the
            different data columns and can be ``ra``, ``dec``, ``weights``, and
            ``redshifts``, where the latter two are optional.
        """
        _, data = read_patch_data(self.data_path)
        return data

    @property
    def coords(self) -> AngularCoordinates:
        """Coordinates in right ascension and declination, in radian."""
        return DataChunk.get_coords(self.load_data())

    @property
    def weights(self) -> NDArray | None:
        """Weights or ``None`` if there are no weights."""
        if not self.has_weights:
            return None
        return DataChunk.getattr(self.load_data(), "weights")

    @property
    def redshifts(self) -> NDArray | None:
        """Redshifts or ``None`` if there are no redshifts."""
        if not self.has_redshifts:
            return None
        return DataChunk.getattr(self.load_data(), "redshifts")
