from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generator, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from yaw.containers import Tclosed, default_closed
from yaw.utils import AngularCoordinates

Tpath = Union[Path, str]


class InconsistentPatchesError(Exception):
    pass


class InconsistentTreesError(Exception):
    pass


def groupby(key_array: NDArray, value_array: NDArray) -> Generator[tuple[Any, NDArray]]:
    idx_sort = np.argsort(key_array)
    keys_sorted = key_array[idx_sort]
    values_sorted = value_array[idx_sort]

    uniques, idx_split = np.unique(keys_sorted, return_index=True)
    yield from zip(uniques, np.split(values_sorted, idx_split[1:]))


def groupby_binning(
    key_array: NDArray,
    value_array: NDArray,
    *,
    binning: NDArray,
    closed: Tclosed = default_closed,
) -> Generator[tuple[NDArray, NDArray]]:
    binning = np.asarray(binning)
    bin_idx = np.digitize(key_array, binning, right=(closed == "right"))

    for i, bin_array in groupby(bin_idx, value_array):
        if 0 < i < len(binning):  # skip values outside of binning range
            yield binning[i - 1 : i + 1], bin_array


class DataChunk:
    __slots__ = ("data", "patch_ids")
    itemtype = "f8"

    def __init__(
        self,
        data: NDArray,
        patch_ids: NDArray[np.int32] | None = None,
    ) -> None:
        self.data = data
        self.set_patch_ids(patch_ids)

    @classmethod
    def from_columns(
        cls,
        ra: NDArray,
        dec: NDArray,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        patch_ids: NDArray | None = None,
        degrees: bool = True,
        chkfinite: bool = False,
    ):
        asarray = np.asarray_chkfinite if chkfinite else np.asarray

        dtype = [("ra", cls.itemtype), ("dec", cls.itemtype)]
        if weights is not None:
            dtype.append(("weights", cls.itemtype))
        if redshifts is not None:
            dtype.append(("redshifts", cls.itemtype))

        data = np.empty(len(ra), dtype=dtype)
        data["ra"] = np.deg2rad(asarray(ra)) if degrees else asarray(ra)
        data["dec"] = np.deg2rad(asarray(dec)) if degrees else asarray(dec)
        if weights is not None:
            data["weights"] = asarray(weights)
        if redshifts is not None:
            data["redshifts"] = asarray(redshifts)

        return cls(data, patch_ids)

    @classmethod
    def from_chunks(cls, chunks: Sequence[DataChunk]) -> DataChunk:
        data = np.concatenate([chunk.data for chunk in chunks])
        if any(chunk.patch_ids is not None for chunk in chunks):
            patch_ids = np.concatenate([chunk.patch_ids for chunk in chunks])
        else:
            patch_ids = None

        return cls(data, patch_ids)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: ArrayLike) -> DataChunk:
        return DataChunk(
            self.data[index],
            self.patch_ids[index] if self.patch_ids is not None else None,
        )

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
        view_2d = self.data.view(self.itemtype).reshape(len(self.data), -1)
        return AngularCoordinates(view_2d[:, :2])

    @property
    def ra(self) -> NDArray:
        return self.data["ra"]

    @property
    def dec(self) -> NDArray:
        return self.data["dec"]

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

    def split(self, num_chunks: int) -> list[DataChunk]:
        splits_data = np.array_split(self.data, num_chunks)

        if self.patch_ids is not None:
            splits_patch_ids = np.array_split(self.patch_ids, num_chunks)
        else:
            splits_patch_ids = [None] * num_chunks

        return [
            DataChunk(data, patch_ids)
            for data, patch_ids in zip(splits_data, splits_patch_ids)
        ]

    def set_patch_ids(self, patch_ids: NDArray | None):
        if patch_ids is not None:
            patch_ids = np.asarray(patch_ids)
            if patch_ids.shape != (len(self),):
                raise ValueError("'patch_ids' has an invalid shape")

            patch_ids = patch_ids.astype(np.int32, casting="same_kind", copy=False)

        self.patch_ids = patch_ids

    def split_patches(self) -> dict[int, DataChunk]:
        if self.patch_ids is None:
            raise ValueError("'patch_ids' not set")

        chunks = {}
        for patch_id, patch_data in groupby(self.patch_ids, self.data):
            chunks[int(patch_id)] = DataChunk(patch_data)

        return chunks


class MockDataFrame(Sequence):  # avoid explicit pandas dependency
    pass


class PatchBase:
    cache_path: Path
    _has_weights: bool
    _has_redshifts: bool

    @property
    def data_path(self) -> Path:
        return self.cache_path / "data.bin"

    @property
    def has_weights(self) -> bool:
        return self._has_weights

    @property
    def has_redshifts(self) -> bool:
        return self._has_redshifts


class CatalogBase:
    cache_directory: Path

    def get_patch_path(self, patch_id: int) -> Path:
        return self.cache_directory / f"patch_{int(patch_id)}"
