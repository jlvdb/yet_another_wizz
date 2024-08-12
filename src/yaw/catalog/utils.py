from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generator, Literal, Type, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from yaw.coordinates import CoordsSky

Tclosed = Literal["left", "right"]
Tjson = TypeVar("Tjson", bound="JsonSerialisable")
Tpath = Union[Path, str]


def groupby_value(
    values: NDArray,
    **optional_arrays: NDArray | None,
) -> Generator[tuple[Any, dict[str, NDArray]], None, None]:
    idx_sort = np.argsort(values)
    values_sorted = values[idx_sort]
    uniques, _idx_split = np.unique(values_sorted, return_index=True)
    idx_split = _idx_split[1:]

    splitted_arrays = {}
    for name, array in optional_arrays.items():
        if array is not None:
            array_sorted = array[idx_sort]
            splitted_arrays[name] = np.split(array_sorted, idx_split)

    for i, value in enumerate(uniques):
        yield value, {name: splits[i] for name, splits in splitted_arrays.items()}


def groupby_binning(
    values: NDArray,
    binning: NDArray,
    closed: Tclosed = "right",
    **optional_arrays: NDArray | None,
) -> Generator[tuple[NDArray, dict[str, NDArray]], None, None]:
    binning = np.asarray(binning)
    bin_idx = np.digitize(values, binning, right=(closed == "right"))

    for i, bin_array in groupby_value(bin_idx, **optional_arrays):
        if 0 < i < len(binning):  # skip values outside of binning range
            yield binning[i - 1 : i + 1], bin_array


def logarithmic_mid(edges: NDArray) -> NDArray:
    log_edges = np.log10(edges)
    log_mids = (log_edges[:-1] + log_edges[1:]) / 2.0
    return 10.0**log_mids


class DataChunk:
    __slots__ = ("coords", "weights", "redshifts", "patch_ids")

    def __init__(
        self,
        coords: CoordsSky,
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
            CoordsSky(coords),
            parser(weights, np.float64),
            parser(redshifts, np.float64),
            parser(patch_ids, np.int32),
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
            coords=CoordsSky.from_coords(chunk.coords for chunk in chunks),
            weights=concat_optional_attr("weights"),
            redshifts=concat_optional_attr("redshifts"),
            patch_ids=concat_optional_attr("patch_ids"),
        )

    def split(self, num_chunks: int) -> list[DataChunk]:
        chunks = [np.array_split(self.coords.data, num_chunks)]
        for attr in ("weights", "redshifts", "patch_ids"):
            values = getattr(self, attr)
            if values is None:
                splits = [None] * num_chunks
            else:
                splits = np.array_split(values, num_chunks)
            chunks.append(splits)

        return [
            DataChunk(CoordsSky(coords), weights, redshifts, patch_ids)
            for coords, weights, redshifts, patch_ids in zip(*chunks)
        ]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: ArrayLike) -> DataChunk:
        return DataChunk(
            coords=self.coords[index],
            weights=self.weights[index] if self.weights is not None else None,
            redshifts=self.redshifts[index] if self.redshifts is not None else None,
            patch_ids=self.patch_ids[index] if self.patch_ids is not None else None,
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
        for patch_id, attr_dict in groupby_value(
            self.patch_ids,
            coords=self.coords,
            weights=self.weights,
            redshifts=self.redshifts,
        ):
            coords = CoordsSky(attr_dict.pop("coords"))
            chunks[int(patch_id)] = DataChunk(coords, **attr_dict)

        return chunks


class JsonSerialisable(ABC):
    @classmethod
    def from_dict(cls: Type[Tjson], kwarg_dict: dict) -> Tjson:
        return cls(**kwarg_dict)

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_file(cls: Type[Tjson], path: Tpath) -> Tjson:
        with Path(path).open() as f:
            kwarg_dict = json.load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Tpath) -> Tjson:
        with Path(path).open(mode="w") as f:
            json.dump(self.to_dict(), f, indent=4)
