from __future__ import annotations

import logging
from abc import abstractmethod
from collections import deque
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence, Sized

import h5py
import numpy as np
import pyarrow as pa
from astropy.io import fits
from pyarrow import Table, parquet

from yaw import parallel
from yaw.logging import long_num_format

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from typing_extensions import Self

    from yaw.randoms import RandomsBase

    TypeDataChunk = TypeVar("TypeDataChunk", NDArray)

CHUNKSIZE = 16_777_216
"""Default chunk size to use, optimised for parallel performance."""

DATA_ATTRS = ("ra", "dec", "weights", "redshifts", "patch_ids")
"""Default chunk size to use, optimised for parallel performance."""

PATCH_ID_DTYPE = "i2"
"""Default data type for patch IDs, larger integer type will likely result in
memory issues with covariance matrix."""

logger = logging.getLogger(__name__)


class DataFrame(Sequence):
    """Dummy type as stand in for pandas DataFrames."""

    pass


def common_len_assert(sized: Iterable[Sized]) -> int:
    """Verify that a set of containers has the same length and return this
    common length."""
    length = None
    for item in sized:
        if length is None:
            length = len(item)
        else:
            if len(item) != length:
                raise ValueError("length of inputs does not match")
    return length


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
            Numpy array with input values stored in fields.

        Raises:
            ValueError:
                If any of the input arrays cannot be casted.
        """
        values = (ra, dec, weights, redshifts, patch_ids)
        inputs = {
            attr: value for attr, value in zip(DATA_ATTRS, values) if value is not None
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
    def pop_field(chunk: TypeDataChunk, field: str) -> tuple[TypeDataChunk, NDArray]:
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
    def has_attr(chunk: TypeDataChunk, attr: str) -> bool:
        return attr in chunk.dtype.fields


class DataChunkReader(AbstractContextManager, Iterator[TypeDataChunk]):
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
    @abstractmethod
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


class RandomReader(DataChunkReader):
    """
    Read a fixed size random sample in chunks from a random generator.

    Can be used in a context manager.

    Args:
        generator:
            Supported random generator, instance of
            :obj:`~yaw.randoms.RandomsBase`.
        num_randoms:
            Total number of samples to read from the generator.
        chunksize:
            Size of each data chunk, optional.
    """

    def __init__(
        self, generator: RandomsBase, num_randoms: int, chunksize: int | None = None
    ) -> None:
        self.generator = generator

        self._num_records = num_randoms
        self.chunksize = chunksize or CHUNKSIZE

        iter(self)  # reset state

        if parallel.on_root():
            logger.info(
                "generating %s random points in %d chunks",
                long_num_format(num_randoms),
                len(self),
            )

    @property
    def has_weights(self) -> bool:
        has_weights = None
        if parallel.on_root():
            has_weights = self.generator.has_weights
        return parallel.COMM.bcast(has_weights, root=0)

    @property
    def has_redshifts(self) -> bool:
        has_redshifts = None
        if parallel.on_root():
            has_redshifts = self.generator.has_redshifts
        return parallel.COMM.bcast(has_redshifts, root=0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        return None

    def _get_next_chunk(self) -> TypeDataChunk:
        # NOTE: _num_samples is already incremented by chunksize in __next__
        probe_size = self.chunksize
        if self._num_samples >= self.num_records:
            probe_size -= self._num_samples - self.num_records

        data = self.generator(probe_size)
        return DataChunk.create(**data, degrees=False, chkfinite=False)

    def __iter__(self) -> Iterator[TypeDataChunk]:
        self.generator.reseed()
        self._num_samples = 0  # reset state
        return self

    def get_probe(self, probe_size: int) -> TypeDataChunk:
        """
        Obtain a (small) sample from the random generator.

        Args:
            probe_size:
                The number of records to generate.

        Returns:
            A chunk of data from the data source with the requested size.

        Raises:
            ValueError:
                If ``probe_size`` exceeds number of records.
        """
        if probe_size > self.num_records:
            raise ValueError("'probe_size' cannot exceed number of records")
        if parallel.on_worker():
            return None

        self.generator.reseed()
        data = self.generator(probe_size)
        return DataChunk.create(**data, degrees=False, chkfinite=False)


class DataReader(DataChunkReader):
    """Base class for reading from a static data source with columnar data."""

    @abstractmethod
    def __init__(
        self,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        columns = (ra_name, dec_name, weight_name, redshift_name, patch_name)
        self._columns = {
            attr: name for attr, name in zip(DATA_ATTRS, columns) if name is not None
        }

        self.degrees = degrees
        self.chunksize = chunksize or CHUNKSIZE

        iter(self)  # reset state

        if parallel.on_root():
            logger.debug("selecting input columns: %s", ", ".join(self.columns))

    @property
    def has_weights(self) -> bool:
        return "weights" in self._columns

    @property
    def has_redshifts(self) -> bool:
        return "redshifts" in self._columns

    def get_probe(self, probe_size: int) -> TypeDataChunk:
        """
        Read a (small) subsample from the data source.

        The returned data is a near-regular subset, with records distributed
        uniformly over the length of the data source.

        Args:
            probe_size:
                The number of records to read.

        Returns:
            A chunk of data from the data source with the requested size.

        Raises:
            ValueError:
                If ``probe_size`` exceeds number of records.
        """
        if parallel.on_root():
            idx_keep = np.linspace(0, self.num_records - 1, probe_size).astype(int)

            chunks = []
            for chunk in iter(self):
                idx_keep = idx_keep[idx_keep >= 0]  # remove previously used indices
                idx_keep_chunk = idx_keep[idx_keep < len(chunk)]  # clip future indices
                idx_keep -= len(chunk)  # shift index list

                chunks.append(chunk[idx_keep_chunk])

            data = np.concatenate(chunks)

        else:
            data = None

        parallel.COMM.Barrier()
        return data


def issue_io_log(num_records: int, num_chunks: int, source: str) -> None:
    """
    Log message issued in __init__ methods of DataReaders.

    Args:
        num_records:
            Number of records that will be processed.
        num_chunks:
            Number of chunks in which the records will be processed.
        source:
            A string describing where the data orginates (e.g. "FITS file").
    """
    if parallel.on_root():
        logger.info(
            "loading %s records in %d chunks from %s",
            long_num_format(num_records),
            num_chunks,
            source,
        )


class DataFrameReader(DataReader):
    """
    Read data in chunks from a :obj:`pandas.DataFrame`.

    Can be used in a context manager.

    Args:
        data:
            Input pandas data frame, containing the columns specified below.
        ra_name:
            Column name of the right ascension coordinate.
        dec_name:
            Column name of the declination coordinate.
        weight_name:
            Optional column name of the object weights.
        redshift_name:
            Optional column name of the object redshifts.
        patch_name:
            Optional column name of patch IDs, must meet patch ID requirements.
        chunksize:
            Size of each data chunk, optional.
        degrees:
            Whether the input coordinates are given in degrees (the default).
    """

    def __init__(
        self,
        data: DataFrame,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        self._data = data
        self._num_records = len(data)
        issue_io_log(self.num_records, self.num_chunks, "memory")

        super().__init__(
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        return None

    def _get_next_chunk(self) -> TypeDataChunk:
        start = self._num_samples
        end = start + self.chunksize
        chunk = self._data[start:end]

        kwargs = {attr: chunk[name].to_numpy() for attr, name in self._columns.items()}
        return DataChunk.create(**kwargs, degrees=self.degrees)


class FileReader(DataReader):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._close_file()

    def _close_file(self) -> None:
        if parallel.on_root():
            self._file.close()


def new_filereader(
    path: Path | str,
    *,
    ra_name: str,
    dec_name: str,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    patch_name: str | None = None,
    chunksize: int | None = None,
    degrees: bool = True,
    **reader_kwargs,
) -> FileReader:
    ext = Path(path).suffix.lower()
    if ext in (".fits", ".cat"):
        reader_cls = FitsReader
    elif ext in (".hdf5", ".hdf", ".h5"):
        reader_cls = HDFReader
    elif ext in (".pq", ".pqt", ".parq", ".parquet"):
        reader_cls = ParquetReader
    else:
        raise ValueError(f"unrecognized file extesion '{ext}'")

    return reader_cls(
        path,
        ra_name=ra_name,
        dec_name=dec_name,
        weight_name=weight_name,
        redshift_name=redshift_name,
        patch_name=patch_name,
        chunksize=chunksize,
        degrees=degrees,
        **reader_kwargs,
    )


class FitsReader(FileReader):
    """
    Read data in chunks from a HDF5 file.

    Must be used in a context manager.

    Args:
        path:
            String or :obj:`pathlib.Path` describing the input file path.
        ra_name:
            Column name of the right ascension coordinate.
        dec_name:
            Column name of the declination coordinate.
        weight_name:
            Optional column name of the object weights.
        redshift_name:
            Optional column name of the object redshifts.
        patch_name:
            Optional column name of patch IDs, must meet patch ID requirements.
        chunksize:
            Size of each data chunk, optional.
        degrees:
            Whether the input coordinates are given in degrees (the default).
        hdu:
            Index of the table HDU to read from (default: 1).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        hdu: int = 1,
    ) -> None:
        self._num_records = None
        if parallel.on_root():
            self._file = fits.open(str(path))
            self._hdu_data = self._file[hdu].data
            self._num_records = len(self._hdu_data)

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        issue_io_log(self.num_records, self.num_chunks, f"FITS file: {self.path}")

        super().__init__(
            path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )

    def _load_next_chunk(self) -> DataChunk:
        def get_data_swapped(colname: str) -> NDArray:
            start = self._num_samples
            end = start + self.chunksize
            array = self._hdu_data[colname][start:end]

            return array.view(array.dtype.newbyteorder()).byteswap()

        kwargs = {attr: get_data_swapped(col) for attr, col in self._columns.items()}
        return DataChunk.create(**kwargs, degrees=self.degrees)


class HDFReader(FileReader):
    """
    Read data in chunks from a HDF5 file.

    Must be used in a context manager.

    Args:
        path:
            String or :obj:`pathlib.Path` describing the input file path.
        ra_name:
            Column name of the right ascension coordinate.
        dec_name:
            Column name of the declination coordinate.
        weight_name:
            Optional column name of the object weights.
        redshift_name:
            Optional column name of the object redshifts.
        patch_name:
            Optional column name of patch IDs, must meet patch ID requirements.
        chunksize:
            Size of each data chunk, optional.
        degrees:
            Whether the input coordinates are given in degrees (the default).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        self._num_records = None
        if parallel.on_root():
            self._file = h5py.File(path, mode="r")
            self._num_records = len(self._file[ra_name])

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        issue_io_log(self.num_records, self.num_chunks, f"HDF5 file: {self.path}")

        super().__init__(
            path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )
        # this final check is necessary since HDF5 columns are independent
        if parallel.on_root():
            common_len_assert([self._file[col] for col in self._columns.values()])

    def _load_next_chunk(self) -> DataChunk:
        start = self._num_samples
        end = start + self.chunksize
        kwargs = {
            attr: self._file[col][start:end] for attr, col in self._columns.items()
        }
        return DataChunk.create(**kwargs, degrees=self.degrees)


class ParquetReader(FileReader):
    """
    Read data in chunks from a Parquet file.

    Data is read in the provided ``chunksize``, bypassing any row groups stored
    in the Parquet file. Must be used in a context manager.

    Args:
        path:
            String or :obj:`pathlib.Path` describing the input file path.
        ra_name:
            Column name of the right ascension coordinate.
        dec_name:
            Column name of the declination coordinate.
        weight_name:
            Optional column name of the object weights.
        redshift_name:
            Optional column name of the object redshifts.
        patch_name:
            Optional column name of patch IDs, must meet patch ID requirements.
        chunksize:
            Size of each data chunk, optional.
        degrees:
            Whether the input coordinates are given in degrees (the default).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        self._num_records = None
        if parallel.on_root():
            self._file = parquet.ParquetFile(self.path)
            self._num_records = self._file.metadata.num_rows

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        issue_io_log(self.num_records, self.num_chunks, f"Parquet file: {self.path}")

        super().__init__(
            path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )

        # this final check is necessary since HDF5 columns are independent
        if parallel.on_root():
            common_len_assert([self._file[col] for col in self._columns.values()])

    def _get_group_cache_size(self) -> int:
        """Get the number of records currently stored in the row-group cache."""
        return sum(len(group) for group in self._group_cache)

    def _load_groups(self) -> None:
        """Keep reading row-groups from the input file until a full chunk can be
        constructed."""
        while self._get_group_cache_size() < self.chunksize:
            group_idx = len(self._group_cache)
            next_group = self._file.read_row_group(group_idx, self._columns.values())
            self._group_cache.append(next_group)

    def _extract_chunk(self) -> Table:
        """Extract a data from the row-group cache and return a chunk as a
        :obj:`pyarrow.Table`."""
        num_records = 0
        groups = []
        while num_records < self.chunksize:
            next_group = self._group_cache.popleft()
            num_records += len(next_group)
            groups.append(next_group)

        oversized_chunk = pa.concat_tables(groups)
        remainder = oversized_chunk[self.chunksize :]
        if len(remainder) > 0:
            self._group_cache.appendleft(remainder)

        return oversized_chunk[: self.chunksize]

    def _load_next_chunk(self) -> DataChunk:
        self._load_groups()
        table = self._extract_chunk()

        kwargs = {
            attr: table.column(col).to_numpy() for attr, col in self._columns.items()
        }
        return DataChunk.create(**kwargs, degrees=self.degrees)

    def __iter__(self) -> Iterator[TypeDataChunk]:
        # additonal iteration state requried by parquet file implementation
        self._group_cache = deque()
        return super().__iter__()
