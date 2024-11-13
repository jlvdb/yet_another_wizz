from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterator, NewType, Sized

import numpy as np
from numpy.typing import NDArray

from yaw.coordinates import AngularCoordinates
from yaw.options import NotSet
from yaw.utils import common_len_assert, parallel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    from numpy.typing import ArrayLike, DTypeLike

    TypePatchIDs = NewType("TypePatchIDs", NDArray[np.int16])
    T = TypeVar("T")
TypeDataChunk = NewType("TypeDataChunk", NDArray)

PATCH_ID_DTYPE = "i2"
"""Default data type for patch IDs, larger integer type will likely result in
memory issues with covariance matrix."""


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
    ATTR_NAMES = ("ra", "dec", "weights", "redshifts", "patch_ids")
    """Default chunk size to use, optimised for parallel performance."""

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
    ) -> TypeDataChunk:
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
        values = (ra, dec, weights, redshifts, patch_ids)
        inputs = {
            attr: value
            for attr, value in zip(DataChunk.ATTR_NAMES, values)
            if value is not None
        }

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

        return array

    @staticmethod
    def get_coords(chunk: TypeDataChunk) -> AngularCoordinates:
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


class DataChunkReader(AbstractContextManager, Sized, Iterator[TypeDataChunk]):
    """
    Base class for chunked data readers and generators.

    Iterates the data source in a fixed chunk size up to a fixed capacity.

    .. Caution::
        This iterator is supposed to work in an MPI environment as follows:
        - On the root worker, generate/load the data and yield it.
        - On a non-root worker yield None. The user is resposible for
          broadcasting if desired.
    """

    chunksize: int
    _num_records: int
    _num_samples: int  # used to track iteration state

    def __len__(self) -> int:
        return self.num_chunks

    @property
    @abstractmethod
    def has_weights(self) -> bool:
        """Whether this data source provides weights."""
        pass

    @property
    @abstractmethod
    def has_redshifts(self) -> bool:
        """Whether this data source provides redshifts."""
        pass

    @property
    def num_records(self) -> int:
        """The number of records this reader produces."""
        return self._num_records

    @property
    def num_chunks(self) -> int:
        """The number of chunks in which the reader produces all records."""
        return int(np.ceil(self.num_records / self.chunksize))

    @abstractmethod
    def get_probe(self, probe_size: int) -> TypeDataChunk:
        """
        Get a (small) subsample from the data source.

        Depending on the source, this may be a randomly generated sample or a
        (alomst) regular subset of data records.

        Args:
            probe_size:
                The number of records to obtain.

        Returns:
            A chunk of data from the data source with the requested size.

        Raises:
            ValueError:
                If ``probe_size`` exceeds number of records.
        """
        pass

    @abstractmethod
    def _get_next_chunk(self) -> TypeDataChunk:
        """Generate or read the next chunk of data from the source."""
        pass

    def __next__(self) -> TypeDataChunk:
        if self._num_samples >= self.num_records:
            raise StopIteration()

        self._num_samples += self.chunksize

        if parallel.on_worker():
            return None
        return self._get_next_chunk()

    def __iter__(self) -> Iterator[TypeDataChunk]:
        self._num_samples = 0  # reset state
        return self
