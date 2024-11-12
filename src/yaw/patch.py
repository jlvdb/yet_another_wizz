from __future__ import annotations

import logging
from itertools import compress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.containers import YamlSerialisable
from yaw.coordinates import AngularCoordinates, AngularDistances
from yaw.readers import DataChunk, TypeDataChunk

if TYPE_CHECKING:
    from io import TextIOBase
    from typing import Any

    from numpy.typing import NDArray

__all__ = [
    "Patch",
]

PATCH_DATA_FILE = "data.bin"

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
        total:
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
        "total",
        "center",
        "radius",
    )

    num_records: int
    """Number of data points in the patch."""
    total: float
    """Sum of point weights."""
    center: AngularCoordinates
    """Center point (mean) of all data points."""
    radius: AngularDistances
    """Radius around center point containing all data points."""

    def __init__(
        self,
        *,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"total={self.total}",
            f"center={self.center}",
            f"radius={self.radius}",
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
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

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
            total=float(self.total),
            center=self.center.tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


def write_header(file: TextIOBase, *, has_weights: bool, has_redshifts: bool) -> None:
    info = (1 << 0) | (1 << 1) | (has_weights << 2) | (has_redshifts << 3)
    info_bytes = info.to_bytes(1, byteorder="big")
    file.write(info_bytes)


def read_header(file: TextIOBase) -> dict[str, bool]:
    header_byte = file.read(1)
    header_int = int.from_bytes(header_byte, byteorder="big")
    return dict(
        ra=bool(header_int & (1 << 0)),
        dec=bool(header_int & (1 << 1)),
        weights=bool(header_int & (1 << 2)),
        redshifts=bool(header_int & (1 << 3)),
    )


def read_patch_data(path: Path | str) -> TypeDataChunk:
    with open(path, mode="rb") as f:
        column_info = read_header(f)
        columns = compress(DataChunk.ATTR_NAMES, column_info.values())
        dtype = np.dtype([(col, "f8") for col in columns])
        rawdata = np.fromfile(f, dtype=np.byte)

    return rawdata.view(dtype)


class PatchWriter:
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
        has_weights: bool,
        has_redshifts: bool,
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
        write_header(self._file, has_weights=has_weights, has_redshifts=has_redshifts)

    def __repr__(self) -> str:
        items = (
            f"buffersize={self.buffersize}",
            f"cachesize={self.cachesize}",
            f"processed={self._num_processed}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_path}"

    @property
    def data_path(self) -> Path:
        """Path to binary file with patch data."""
        return self.cache_path / PATCH_DATA_FILE

    @property
    def cachesize(self) -> int:
        return sum(len(shard) for shard in self._shards)

    def open(self) -> None:
        if self._file is None:
            self._file = self.data_path.open(mode="ab")

    def flush(self) -> None:
        if len(self._shards) > 0:
            self.open()  # ensure file is ready for writing

            data = np.concatenate(self._shards)
            self._num_processed += len(data)
            self._shards = []

            data.tofile(self._file)

    def close(self) -> None:
        self.flush()
        self._file.close()
        self._file = None

    def process_chunk(self, data: TypeDataChunk) -> None:
        if DataChunk.hasattr(data, "patch_ids"):
            raise ValueError("'patch_ids' field must stripped before writing data")

        self._shards.append(data)

        if self.cachesize >= self.buffersize:
            self.flush()


class Patch:
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
          ├╴ binning   (optional, see trees.BinnedTrees)
          └╴ trees.pkl (optional, see trees.BinnedTrees)

    Supports efficient pickeling as long as the cached data is not deleted or
    moved.
    """

    __slots__ = ("meta", "cache_path", "_has_weights", "_has_redshifts")

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
                self._has_weights, self._has_redshifts = read_header(f)

        except FileNotFoundError:
            data = read_patch_data(self.data_path)
            self._has_weights = DataChunk.hasattr(data, "weights")
            self._has_redshifts = DataChunk.hasattr(data, "redshifts")

            self.meta = Metadata.compute(
                DataChunk.get_coords(data),
                weights=DataChunk.getattr(data, "weights", None),
                center=center,
            )
            self.meta.to_file(meta_data_file)

    def __repr__(self) -> str:
        items = (
            f"num_records={self.meta.num_records}",
            f"total={self.meta.total}",
            f"has_weights={self._has_weights}",
            f"has_redshifts={self._has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_path}"

    def __getstate__(self) -> dict:
        return dict(
            cache_path=self.cache_path,
            meta=self.meta,
            _has_weights=self._has_weights,
            _has_redshifts=self._has_redshifts,
        )

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    @property
    def data_path(self) -> Path:
        """Path to binary file with patch data."""
        return self.cache_path / PATCH_DATA_FILE

    @property
    def has_weights(self) -> bool:
        """Whether the patch data contain weights."""
        return self._has_weights

    @property
    def has_redshifts(self) -> bool:
        """Whether the patch data contain redshifts."""
        return self._has_redshifts

    def load_data(self) -> TypeDataChunk:
        """
        Load the cached object data with coordinates and optional weights and
        redshifts.

        Returns:
            A special :obj:`PatchData` container that has the same
            :obj:`coords`, :obj:`weights`, and :obj:`redshifts` attributes as
            :obj:`Patch`.
        """
        return read_patch_data(self.data_path)

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
