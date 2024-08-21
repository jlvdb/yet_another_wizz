from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generator, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from yaw.containers import Serialisable, Tclosed, default_closed
from yaw.coordinates import AngularCoordinates

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


class DataChunk(Serialisable):
    __slots__ = ("coords", "weights", "redshifts", "patch_ids")

    def __init__(
        self,
        coords: AngularCoordinates,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        patch_ids: NDArray[np.int32] | None = None,
    ) -> None:
        self.coords = coords
        self.weights = weights
        self.redshifts = redshifts
        self.set_patch_ids(patch_ids)

    @classmethod
    def from_columns(
        cls,
        ra: NDArray,
        dec: NDArray,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        patch_ids: NDArray | None = None,
        degrees: bool = True,
        chkfinite: bool = False,
    ):
        def parser(arr: NDArray | None, dtype: DTypeLike) -> NDArray | None:
            if arr is None:
                return None
            if chkfinite:
                return np.asarray_chkfinite(arr, dtype=dtype)
            return arr.astype(dtype, casting="same_kind", copy=False)

        ra = parser(ra, np.float64)
        dec = parser(dec, np.float64)
        coords = np.column_stack((ra, dec))
        if degrees:
            coords = np.deg2rad(coords)

        return cls(
            coords=AngularCoordinates(coords),
            weights=parser(weights, np.float64),
            redshifts=parser(redshifts, np.float64),
            patch_ids=parser(patch_ids, np.int32),
        )

    @classmethod
    def from_chunks(cls, chunks: Sequence[DataChunk]) -> DataChunk:
        def concat_optional_attr(attr: str) -> NDArray | None:
            values = tuple(getattr(chunk, attr) for chunk in chunks)
            value_is_set = tuple(value is not None for value in values)
            if all(value_is_set):
                return np.concatenate(values)
            elif not any(value_is_set):
                return None
            raise ValueError(f"not all chunks have '{attr}' set")

        return DataChunk(
            coords=AngularCoordinates.from_coords(chunk.coords for chunk in chunks),
            weights=concat_optional_attr("weights"),
            redshifts=concat_optional_attr("redshifts"),
            patch_ids=concat_optional_attr("patch_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        the_dict = dict(coords=self.coords)
        if self.weights is not None:
            the_dict["weights"] = self.weights
        if self.redshifts is not None:
            the_dict["redshifts"] = self.redshifts
        if self.patch_ids is not None:
            the_dict["patch_ids"] = self.patch_ids
        return the_dict

    def split(self, num_chunks: int) -> list[DataChunk]:
        splitted = dict(
            coords=[
                AngularCoordinates(v) for v in np.array_split(self.coords, num_chunks)
            ]
        )
        for attr, values in self.to_dict().items():
            splits = np.array_split(values, num_chunks)
            if attr == "coords":
                splits = [AngularCoordinates(split) for split in splits]
            splitted[attr] = splits

        return [
            DataChunk.from_dict({attr: values[i] for attr, values in splitted.items()})
            for i in range(num_chunks)
        ]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: ArrayLike) -> DataChunk:
        return DataChunk.from_dict(
            {attr: values[index] for attr, values in self.to_dict().items()}
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
            raise ValueError("'patch_ids' not provided")

        chunks = {}
        for patch_id, attr_dict in groupby_value(self.patch_ids, **self.to_dict()):
            coords = AngularCoordinates(attr_dict.pop("coords"))
            chunks[int(patch_id)] = DataChunk(coords, **attr_dict)

        return chunks


class MockDataFrame(Sequence):  # avoid explicit pandas dependency
    pass
