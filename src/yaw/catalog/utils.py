from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generator, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from yaw.containers import Tclosed, default_closed
from yaw.utils import AngularCoordinates

__all__ = [
    "DataChunk",
    "groupby_value",
    "groupby_binning",
]

Tpath = Union[Path, str]


def groupby_value(
    values: NDArray,
    **arrays: NDArray,
) -> Generator[tuple[Any, dict[str, NDArray]], None, None]:
    idx_sort = np.argsort(values)
    values_sorted = values[idx_sort]
    uniques, _idx_split = np.unique(values_sorted, return_index=True)

    idx_split = _idx_split[1:]
    splitted_arrays = {
        name: np.split(array[idx_sort], idx_split) for name, array in arrays.items()
    }

    for i, value in enumerate(uniques):
        yield value, {name: splits[i] for name, splits in splitted_arrays.items()}


def groupby_binning(
    values: NDArray,
    binning: NDArray,
    closed: Tclosed = default_closed,
    **arrays: NDArray,
) -> Generator[tuple[NDArray, dict[str, NDArray]], None, None]:
    binning = np.asarray(binning)
    bin_idx = np.digitize(values, binning, right=(closed == "right"))

    for i, bin_array in groupby_value(bin_idx, **arrays):
        if 0 < i < len(binning):  # skip values outside of binning range
            yield binning[i - 1 : i + 1], bin_array


class DataChunk:
    __slots__ = ("data", "patch_ids")

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
    ):

        dtype = [("ra", "f8"), ("dec", "f8")]
        if weights is not None:
            dtype.append(("weights", "f8"))
        if redshifts is not None:
            dtype.append(("redshifts", "f8"))

        data = np.empty(len(ra), dtype=dtype)
        data["ra"] = np.deg2rad(ra) if degrees else ra
        data["dec"] = np.deg2rad(dec) if degrees else dec
        if weights is not None:
            data["weights"] = weights
        if redshifts is not None:
            data["redshifts"] = redshifts

        return cls(data, patch_ids)

    @classmethod
    def from_chunks(cls, chunks: Sequence[DataChunk]) -> DataChunk:
        data = np.concatenate([chunk.data for chunk in chunks])
        if any(chunk.patch_ids is not None for chunk in chunks):
            patch_ids = np.concatenate([chunk.patch_ids for chunk in chunks])

        return cls(data, patch_ids)

    def view(self) -> NDArray:
        num_cols = len(self.data.dtype.fields)
        return self.data.view("f8").reshape(-1, num_cols)

    @property
    def coords(self) -> AngularCoordinates:
        return AngularCoordinates(self.view()[:, :2])

    @property
    def weights(self) -> NDArray | None:
        if "weights" in self.data.dtype.fields:
            return self.view()[:, 2]
        return None

    @property
    def redshifts(self) -> NDArray | None:
        if "redshifts" in self.data.dtype.fields:
            idx_col = 2 + ("weights" in self.data.dtype.fields)
            return self.view()[:, idx_col]
        return None

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: ArrayLike) -> DataChunk:
        return DataChunk(
            self.data[index],
            self.patch_ids[index] if self.patch_ids is not None else None,
        )

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
        for patch_id, data_dict in groupby_value(self.patch_ids, data=self.data):
            chunks[int(patch_id)] = DataChunk(**data_dict)

        return chunks


class MockDataFrame(Sequence):  # avoid explicit pandas dependency
    pass
