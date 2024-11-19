"""
Implements helper functions and base classes for objects that handle chunks of
catalog data.

Data chunks are numpy arrays with composite data type (named fields) and the
helper functions simplify manipulating them. The data classes implement an
interface that simplifies passing around which of the optional data attributes
(weights, redshifts, ...) are contained in a data chunk.
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, NewType

import numpy as np
from numpy.typing import NDArray

from yaw.coordinates import AngularCoordinates
from yaw.options import NotSet
from yaw.utils import common_len_assert

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    from numpy.typing import ArrayLike, DTypeLike

    TypePatchIDs = NewType("TypePatchIDs", NDArray[np.int16])
    T = TypeVar("T")
TypeDataChunk = NewType("TypeDataChunk", NDArray)

__all__ = [
    "ATTR_ORDER",
    "DataChunkInfo",
    "DataChunk",
]


PATCH_ID_DTYPE = "i2"
"""Default data type for patch IDs, larger integer type will likely result in
memory issues with covariance matrix."""

ATTR_ORDER = ("ra", "dec", "weights", "redshifts", "patch_ids")
"""The order of attributes in a DataChunk."""


@dataclass
class DataChunkInfo:
    """
    Helper class to specify which of the (optional) data chunk attributes are
    available.

    Args:
        has_weights:
            Whether weights are available.
        has_redshifts:
            Whether redshifts are available.
        has_patch_ids:
            Whether patch IDs are available.
    """

    # match to ATTR_ORDER
    has_weights: bool = field(default=False)
    has_redshifts: bool = field(default=False)
    has_patch_ids: bool = field(default=False)

    @classmethod
    def from_bytes(cls, info_bytes: bytes) -> DataChunkInfo:
        """
        Restore a class instance from a single byte of bit flags.

        Args:
            info_bytes:
                The byte(s) encoding the state information.

        Returns:
            A new class instance.
        """
        state = int.from_bytes(info_bytes, byteorder="big")
        return cls(  # match to ATTR_ORDER
            has_weights=bool(state & (1 << 2)),
            has_redshifts=bool(state & (1 << 3)),
            has_patch_ids=bool(state & (1 << 4)),
        )

    def to_bytes(self) -> bytes:
        """
        Represent this instance as bit flags stored in a single byte.

        Creates a single big endian byte, where the bits indicate which of the
        attributes in :obj:`ATTR_ORDER` are available.

        Returns:
            A single byte with bit flags.
        """
        info = (  # match to ATTR_ORDER
            (True << 0)  # "ra"
            | (True << 1)  # "dec"
            | (self.has_weights << 2)
            | (self.has_redshifts << 3)
            | (self.has_patch_ids << 4)
        )
        return info.to_bytes(1, byteorder="big")

    def get_list(self) -> list[str]:
        attrs = [attr for attr in ATTR_ORDER[:2]]
        attrs.extend(attr for attr in ATTR_ORDER[2:] if getattr(self, f"has_{attr}"))
        return attrs

    def format(self, *, skip_patch_ids: bool = True) -> str:
        """Helper function to format the boolean attribute flags."""
        values = asdict(self).copy()
        if skip_patch_ids:
            values.pop("has_patch_ids", None)
        return ", ".join(f"{attr}={value}" for attr, value in values.items())


class HandlesDataChunk(ABC):
    """
    Base class of objects that create or process data chunks.
    """

    _chunk_info: DataChunkInfo

    @property
    def has_weights(self) -> bool:
        """Whether this data source provides weights."""
        return self._chunk_info.has_weights

    @property
    def has_redshifts(self) -> bool:
        """Whether this data source provides redshifts."""
        return self._chunk_info.has_redshifts

    @property
    def has_patch_ids(self) -> bool:
        """Whether this data source provides patch IDs."""
        return self._chunk_info.has_patch_ids

    def copy_chunk_info(self) -> DataChunkInfo:
        """Copy the data attribute information."""
        return deepcopy(self._chunk_info)


def get_array_dtype(column_names: Iterable[str]) -> DTypeLike:
    """
    Construct a composite data type for a numpy array to hold a chunk of data.

    By default, all fields will default to 64 bit floats, patch IDs to the
    default integer type.
    """
    special_types = dict(patch_ids=PATCH_ID_DTYPE)
    default_type = "f8"
    array_dtype = [
        (name, special_types.get(name, default_type)) for name in column_names
    ]
    return np.dtype(array_dtype)


def check_patch_ids(patch_ids: ArrayLike) -> None:
    """Ensure that an array-like contains values that fit into the default data
    type for patch IDs."""
    patch_ids = np.asarray(patch_ids)

    min_id = 0
    max_id = np.iinfo(PATCH_ID_DTYPE).max
    if patch_ids.min() < min_id or patch_ids.max() > max_id:
        raise ValueError(f"'patch_ids' must be in range [{min_id}, {max_id}]")


class DataChunk:
    """
    Collection of functions to handle chunks of catalog data.

    For simplicity, the chunks of data are just numpy arrays with a composite
    type (named fields) instead of a dedicated class. The advantage is, that the
    data is stored in a contiguous memory chunk that can be read from or written
    to disk directly.
    """

    @staticmethod
    def create(
        ra: NDArray,
        dec: NDArray,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        patch_ids: NDArray | None = None,
        degrees: bool = True,
        chkfinite: bool = True,
    ) -> tuple[DataChunkInfo, TypeDataChunk]:
        """
        Create a new numpy array holding a chunk of data.

        The array has a composite data type with fields for each data column.
        Types are casted automatically, values can be check to be finite, and
        coordinates converted to radian.

        Args:
            ra:
                Array of right ascension coordinates.
            dec:
                Array of declination coordinates.

        Keyword Args:
            weights:
                Array of object weights, optional.
            redshifts:
                Array of object redshifts, optional.
            patch_ids:
                Array of patch IDs, must fit into 16-bit integer type, optional.
            degrees:
                Whether the input coordinates are given in degrees (the default).
            chkfinite:
                Whether to ensure that all input values are finite.

        Returns:
            Numpy array with input values stored in fields with order ``ra``,
            ``dec``, (``weights``, ``redshifts``, ``patch_ids``).

        Raises:
            ValueError:
                If any of the input arrays cannot be casted.
        """
        values = (ra, dec, weights, redshifts, patch_ids)  # match to ATTR_ORDER
        inputs = {
            attr: value for attr, value in zip(ATTR_ORDER, values) if value is not None
        }
        data_attrs = DataChunkInfo(
            has_weights=weights is not None,
            has_redshifts=redshifts is not None,
            has_patch_ids=patch_ids is not None,
        )

        if patch_ids is not None:
            check_patch_ids(patch_ids)

        num_records = common_len_assert(inputs.values())
        dtype = get_array_dtype(inputs.keys())
        array = np.empty(num_records, dtype=dtype)

        asarray_func = np.asarray_chkfinite if chkfinite else np.asarray
        for name, value in inputs.items():
            array[name] = asarray_func(value)

        if degrees:
            array["ra"] = np.deg2rad(array["ra"])
            array["dec"] = np.deg2rad(array["dec"])

        return data_attrs, array

    @staticmethod
    def get_coords(chunk: TypeDataChunk) -> AngularCoordinates:
        """Receive the stored coordinates as :ob:`AngularCoordinates`."""
        coords = np.empty((len(chunk), 2), dtype="f8")
        coords[:, 0] = chunk["ra"]
        coords[:, 1] = chunk["dec"]
        return AngularCoordinates(coords)

    @staticmethod
    def pop(chunk: TypeDataChunk, field: str) -> tuple[TypeDataChunk, NDArray]:
        """
        Remove a field from a numpy array holding a chunk of data and return
        its values.

        Args:
            chunk:
                Numpy array with composite data type (fields).
            field:
                Name of the field to remove from the chunk.

        Returns:
            The chunk without the specified field/column and an array with the
            removed values.
        """
        popped_values = chunk[field]

        dtype = {name: dtype for name, (dtype, offset) in chunk.dtype.fields.items()}
        dtype.pop(field)

        new_chunk = np.array(len(chunk), dtype=np.dtype(dtype))
        for name in dtype:
            new_chunk[name] = chunk[name]
        return new_chunk, popped_values

    @staticmethod
    def hasattr(chunk: TypeDataChunk, attr: str) -> bool:
        """
        Check if a numpy array holding a chunk of data contains a given field.

        Args:
            chunk:
                Numpy array with composite data type (fields).
            field:
                Name of the field to remove from the chunk.

        Returns:
            Whether the field is contained in the array.
        """
        return attr in chunk.dtype.fields

    @staticmethod
    def getattr(chunk: TypeDataChunk, attr: str, default: T = NotSet) -> NDArray | T:
        """
        Retrieve a given field from a numpy array holding a chunk of data.

        Args:
            chunk:
                Numpy array with composite data type (fields).
            field:
                Name of the field to remove from the chunk.
            default:
                Optional value to return in case the array does not contain the
                field.

        Returns:
            The values stored in the field or the default if the field is not
            present.

        Raises:
            ValueError:
                If the array does not contain the field and no default is
                provided.
        """
        try:
            return chunk[attr]
        except ValueError:
            if default is NotSet:
                raise
            return default
