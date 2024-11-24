"""
Implements reader classes that read input files or draws a fixed-size random
sample from a generator.

Readers serve as data source when creating the catalogs used for correlation
measurements. The input data source is processed in chunks to allow out-of-
memory processing of large datasets.
"""

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
from pyarrow import ArrowException, Table, parquet

from yaw.datachunk import (
    ATTR_ORDER,
    DataChunk,
    DataChunkInfo,
    HandlesDataChunk,
    TypeDataChunk,
)
from yaw.utils import common_len_assert, format_long_num, parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.randoms import RandomsBase

__all__ = [
    "DataFrameReader",
    "FitsReader",
    "HDFReader",
    "ParquetReader",
    "RandomReader",
    "new_filereader",
]

CHUNKSIZE = 16_777_216
"""Default chunk size to use, optimised for parallel performance."""

logger = logging.getLogger(__name__)


class DataFrame(Sequence):
    """Dummy type as stand in for pandas DataFrames."""

    pass


class DataChunkReader(
    AbstractContextManager, Sized, Iterator[TypeDataChunk], HandlesDataChunk
):
    """
    Base class for reading data in chunks from a data source.

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

    def _reset_iter_state(self) -> None:
        self._num_samples = 0

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
        self._reset_iter_state()
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
        self._chunk_info = generator.copy_chunk_info()

        self._num_records = num_randoms
        self.chunksize = chunksize or CHUNKSIZE

        self._reset_iter_state()

        if parallel.on_root():
            logger.info(
                "generating %s random points in %d chunks",
                format_long_num(num_randoms),
                len(self),
            )

    def __repr__(self) -> str:
        source = type(self.generator).__name__
        attrs = self._chunk_info.format()
        return f"{type(self)}({source=}, {attrs})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        return None

    def _reset_iter_state(self) -> None:
        super()._reset_iter_state()
        self.generator.reseed()

    def _get_next_chunk(self) -> TypeDataChunk:
        # NOTE: _num_samples is already incremented by chunksize in __next__
        probe_size = self.chunksize
        if self._num_samples >= self.num_records:
            probe_size -= self._num_samples - self.num_records

        return self.generator(probe_size)

    def __iter__(self) -> Iterator[TypeDataChunk]:
        self._reset_iter_state()
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
        return self.generator(probe_size)


class DataReader(DataChunkReader):
    """Base class for reading from a static data source with columnar data."""

    @abstractmethod
    def __init__(
        self,
        *args,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        columns = (
            ra_name,
            dec_name,
            weight_name,
            redshift_name,
            patch_name,
        )  # match to ATTR_ORDER
        self._columns = {
            attr: name for attr, name in zip(ATTR_ORDER, columns) if name is not None
        }
        self._chunk_info = DataChunkInfo(
            has_weights=weight_name is not None,
            has_redshifts=redshift_name is not None,
            has_patch_ids=patch_name is not None,
        )

        self.degrees = degrees
        self.chunksize = chunksize or CHUNKSIZE

        self._reset_iter_state()

        if parallel.on_root():
            logger.debug(
                "selecting input columns: %s",
                ", ".join(self._columns.values()),
            )

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
            format_long_num(num_records),
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
        self.chunksize = chunksize or CHUNKSIZE  # we need this early
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

    def __repr__(self) -> str:
        attrs = self._chunk_info.format()
        return f"{type(self)}({attrs})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        return None

    def _get_next_chunk(self) -> TypeDataChunk:
        start = self._num_samples
        end = start + self.chunksize
        chunk = self._data[start:end]

        kwargs = {attr: chunk[name].to_numpy() for attr, name in self._columns.items()}
        _, chunk = DataChunk.create(**kwargs, degrees=self.degrees)
        return chunk


class FileReader(DataReader):
    path: Path

    def __repr__(self) -> str:
        source = str(self.path)
        attrs = self._chunk_info.format()
        return f"{type(self)}({source=}, {attrs})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._close_file()

    def _close_file(self) -> None:
        """Close the underlying file(pointer)."""
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
        self.path = Path(path)
        self._num_records = None
        if parallel.on_root():
            self._file = fits.open(str(path))
            self._hdu_data = self._file[hdu].data
            self._num_records = len(self._hdu_data)

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        self.chunksize = min(self._num_records, chunksize or CHUNKSIZE)
        issue_io_log(self.num_records, self.num_chunks, f"FITS file: {path}")

        super().__init__(
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )

    def _get_next_chunk(self) -> DataChunk:
        def get_data_swapped(colname: str) -> NDArray:
            end = self._num_samples  # already incremented by chunksize in __next__
            start = end - self.chunksize
            array = self._hdu_data[colname][start:end]

            return array.view(array.dtype.newbyteorder()).byteswap()

        kwargs = {attr: get_data_swapped(col) for attr, col in self._columns.items()}
        _, chunk = DataChunk.create(**kwargs, degrees=self.degrees)
        return chunk


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
        self.path = Path(path)
        self._num_records = None
        if parallel.on_root():
            self._file = h5py.File(str(path), mode="r")
            self._num_records = len(self._file[ra_name])

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        self.chunksize = min(self._num_records, chunksize or CHUNKSIZE)
        issue_io_log(self.num_records, self.num_chunks, f"HDF5 file: {path}")

        super().__init__(
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

    def _get_next_chunk(self) -> DataChunk:
        end = self._num_samples  # already incremented by chunksize in __next__
        start = end - self.chunksize
        kwargs = {
            attr: self._file[col][start:end] for attr, col in self._columns.items()
        }
        _, chunk = DataChunk.create(**kwargs, degrees=self.degrees)
        return chunk


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
        self.path = Path(path)
        self._num_records = None
        if parallel.on_root():
            self._file = parquet.ParquetFile(str(path))
            self._num_records = self._file.metadata.num_rows

        self._num_records = parallel.COMM.bcast(self._num_records, root=0)
        self.chunksize = min(self._num_records, chunksize or CHUNKSIZE)
        issue_io_log(self.num_records, self.num_chunks, f"Parquet file: {path}")

        super().__init__(
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
        )

    def _reset_iter_state(self) -> None:
        super()._reset_iter_state()
        self._group_cache = deque()
        self._group_idx = 0  # parquet file iteration state

    def _get_group_cache_size(self) -> int:
        """Get the number of records currently stored in the row-group cache."""
        return sum(len(group) for group in self._group_cache)

    def _load_groups(self) -> None:
        """Keep reading row-groups from the input file until a full chunk can be
        constructed or the end of the file is reached."""
        while self._get_group_cache_size() < self.chunksize:
            try:
                next_group = self._file.read_row_group(
                    self._group_idx, self._columns.values()
                )
                self._group_cache.append(next_group)
                self._group_idx += 1
            except ArrowException:
                break  # end of file reached before chunk is full

    def _extract_chunk(self) -> Table:
        """Extract a data from the row-group cache and return a chunk as a
        :obj:`pyarrow.Table`."""
        num_records = 0
        groups = []
        while num_records < self.chunksize:
            try:
                next_group = self._group_cache.popleft()
                num_records += len(next_group)
                groups.append(next_group)
            except IndexError:
                break  # end of file reached before chunk is full

        oversized_chunk = pa.concat_tables(groups)
        remainder = oversized_chunk[self.chunksize :]
        if len(remainder) > 0:
            self._group_cache.appendleft(remainder)

        return oversized_chunk[: self.chunksize]

    def _get_next_chunk(self) -> DataChunk:
        self._load_groups()
        table = self._extract_chunk()

        kwargs = {
            attr: table.column(col).to_numpy() for attr, col in self._columns.items()
        }
        _, chunk = DataChunk.create(**kwargs, degrees=self.degrees)
        return chunk
