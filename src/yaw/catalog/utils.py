from __future__ import annotations

from collections.abc import Sequence
from itertools import compress
from typing import TYPE_CHECKING, get_args

import numpy as np

from yaw.containers import Tpath
from yaw.utils import AngularCoordinates

if TYPE_CHECKING:
    from io import TextIOBase
    from pathlib import Path
    from typing import Any, Generator, NewType

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    Tpids = NewType("Tpids", NDArray[np.int16])


DATA_ATTRIBUTES = ("ra", "dec", "weights", "redshifts")
PATCH_NAME_TEMPLATE = "patch_{:d}"


def groupby(key_array: NDArray, value_array: NDArray) -> Generator[tuple[Any, NDArray]]:
    idx_sort = np.argsort(key_array)
    keys_sorted = key_array[idx_sort]
    values_sorted = value_array[idx_sort]

    uniques, idx_split = np.unique(keys_sorted, return_index=True)
    yield from zip(uniques, np.split(values_sorted, idx_split[1:]))


class InconsistentPatchesError(Exception):
    pass


class InconsistentTreesError(Exception):
    pass


class PatchIDs:
    itemtype = "i2"

    @staticmethod
    def validate(patch_ids: ArrayLike) -> None:
        min_id = 0
        max_id = np.iinfo(np.int16).max

        patch_ids = np.asarray(patch_ids)
        if patch_ids.ndim > 1:
            raise ValueError("'patch_ids' must be scalar or 1-dimensional")

        if patch_ids.min() < min_id or patch_ids.max() > max_id:
            raise ValueError(f"'patch_ids' must be in range [{min_id}, {max_id}]")

    @staticmethod
    def parse(patch_ids: ArrayLike, num_expect: int = -1) -> Tpids:
        patch_ids = np.atleast_1d(patch_ids)

        PatchIDs.validate(patch_ids)
        if num_expect > 0 and len(patch_ids) != num_expect:
            raise ValueError("'patch_ids' does not have expected length")

        return patch_ids.astype(PatchIDs.itemtype, casting="same_kind", copy=False)


class PatchData:
    __slots__ = ("data",)
    itemtype = "f8"

    def __init__(self, data: NDArray) -> None:
        self.data = data

    @classmethod
    def from_columns(
        cls,
        ra: NDArray,
        dec: NDArray,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        degrees: bool = True,
        chkfinite: bool = True,
    ) -> PatchData:
        columns = compress(
            DATA_ATTRIBUTES, (True, True, weights is not None, redshifts is not None)
        )
        dtype = np.dtype([(col, cls.itemtype) for col in columns])
        asarray = np.asarray_chkfinite if chkfinite else np.asarray

        data = np.empty(len(ra), dtype=dtype)
        data["ra"] = np.deg2rad(asarray(ra)) if degrees else asarray(ra)
        data["dec"] = np.deg2rad(asarray(dec)) if degrees else asarray(dec)
        if weights is not None:
            data["weights"] = asarray(weights)
        if redshifts is not None:
            data["redshifts"] = asarray(redshifts)

        return cls(data)

    @classmethod
    def from_file(cls, file: Tpath | TextIOBase) -> PatchData:
        if isinstance(file, get_args(Tpath)):
            file = open(file, mode="rb")
        with file:
            has_weights, has_redshifts = cls.read_header(file)
            columns = compress(
                DATA_ATTRIBUTES, (True, True, has_weights, has_redshifts)
            )
            dtype = np.dtype([(col, cls.itemtype) for col in columns])

            rawdata = np.fromfile(file, dtype=np.byte)
        return cls(rawdata.view(dtype))

    @staticmethod
    def read_header(file: TextIOBase) -> tuple[bool, bool]:
        header_byte = file.read(1)
        header_int = int.from_bytes(header_byte, byteorder="big")

        has_weights = bool(header_int & (1 << 2))
        has_redshifts = bool(header_int & (1 << 3))
        return has_weights, has_redshifts

    def to_file(self, file: Tpath | TextIOBase) -> None:
        if isinstance(file, get_args(Tpath)):
            file = open(file, mode="wb")
        with file:
            self.write_header(
                file,
                has_weights=self.has_weights,
                has_redshifts=self.has_redshifts,
            )
            self.data.tofile(file)

    @staticmethod
    def write_header(
        file: TextIOBase, *, has_weights: bool, has_redshifts: bool
    ) -> None:
        info = (1 << 0) | (1 << 1) | (has_weights << 2) | (has_redshifts << 3)
        info_bytes = info.to_bytes(1, byteorder="big")
        file.write(info_bytes)

    def __repr__(self) -> str:
        items = (
            f"num_records={len(self)}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    def __len__(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def has_weights(self) -> bool:
        return "weights" in self.data.dtype.fields

    @property
    def has_redshifts(self) -> bool:
        return "redshifts" in self.data.dtype.fields

    @property
    def coords(self) -> AngularCoordinates:
        view = self.data.view(self.itemtype)
        view_2d = view.reshape((len(self.data), -1))

        coords = AngularCoordinates.__new__(AngularCoordinates)
        coords.data = view_2d[:, :2]  # skip checks in AngularCoordinates.__init__
        return coords

    @property
    def weights(self) -> NDArray | None:
        if not self.has_weights:
            return None

        return self.data["weights"]

    @property
    def redshifts(self) -> NDArray | None:
        if not self.has_redshifts:
            return None

        return self.data["redshifts"]


class MockDataFrame(Sequence):  # avoid explicit pandas dependency
    pass


class PatchBase:
    cache_path: Path
    """Directory where (meta) data is cached."""
    _has_weights: bool
    _has_redshifts: bool

    @property
    def data_path(self) -> Path:
        """Path to binary file with patch data."""
        return self.cache_path / "data.bin"

    @property
    def has_weights(self) -> bool:
        """Whether the patch data contain weights."""
        return self._has_weights

    @property
    def has_redshifts(self) -> bool:
        """Whether the patch data contain redshifts."""
        return self._has_redshifts


class CatalogBase:
    cache_directory: Path
    """Directory in which the data is cached in spatial patches."""

    def get_patch_path(self, patch_id: int) -> Path:
        """
        Get the patch to a specific patch cache directory, given the patch
        ID/index.

        Returns:
            Path as a :obj:`pathlib.Path`.
        """
        return self.cache_directory / PATCH_NAME_TEMPLATE.format(patch_id)
