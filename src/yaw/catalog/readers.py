from __future__ import annotations

import logging
import os
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sized
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
from astropy.io import fits
from numpy.typing import NDArray
from pyarrow import Table, parquet
from typing_extensions import Self

from yaw.catalog.utils import MockDataFrame as DataFrame
from yaw.containers import Tpath

__all__ = [
    "DataFrameReader",
    "FitsReader",
    "HDFReader",
    "ParquetReader",
    "new_filereader",
]

Tchunk = dict[str, NDArray]

CHUNKSIZE = 16_777_216

logger = logging.getLogger(__name__)


def long_num_format(x: float | int) -> str:
    """Format a floating point number as string with a numerical suffix.

    E.g.: 1234.0 is converted to ``1.24K``.
    """
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1000:
        exp += 1
        x /= 1000.0
    prefix = str(x).rstrip("0").rstrip(".")
    suffix = ["", "K", "M", "B", "T"][exp]
    return prefix + suffix


def swap_byteorder(array: NDArray) -> NDArray:
    return array.view(array.dtype.newbyteorder()).byteswap()


class BaseReader(Sized, Iterator[Tchunk], AbstractContextManager):
    @abstractmethod
    def __init__(
        self,
        source: Any,
        *,
        columns: Iterable,
        chunksize: int | None = None,
        **reader_kwargs,
    ) -> None:
        self.columns = list(columns)

        self.chunksize = chunksize or CHUNKSIZE
        self._chunk_idx = 0

        self._init_source(source, **reader_kwargs)

        logger.debug("selecting input columns: %s", ", ".join(self.columns))

    @abstractmethod
    def _init_source(self, source: Any, **reader_kwargs) -> None:
        self._num_chunks = int(np.ceil(self.num_records / self.chunksize))

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name} @ {self._chunk_idx} / {self.num_chunks} chunks"

    @property
    @abstractmethod
    def num_records(self) -> int:
        pass

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @abstractmethod
    def _load_next_chunk(self) -> Tchunk:
        pass

    def __len__(self) -> int:
        return self.num_chunks

    def __next__(self) -> Tchunk:
        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        chunk = self._load_next_chunk()
        self._chunk_idx += 1
        return chunk

    def __iter__(self) -> Iterator[Tchunk]:
        self._chunk_idx = 0
        return self

    @abstractmethod
    def read(self, sparse: int) -> Tchunk:
        n_read = 0
        chunks: dict[str, list[NDArray]] = defaultdict(list)
        for chunk in self:
            # keep track of where the next spare record in the new chunk is located
            chunk_offset = ((n_read // sparse + 1) * sparse - n_read) % sparse
            chunk_size = len(next(iter(chunk.values())))

            sparse_idx = np.arange(chunk_offset, chunk_size, sparse)
            n_read += chunk_size

            for colname, array in chunk.items():
                chunks[colname].append(array[sparse_idx])

        return {colname: np.concatenate(arrays) for colname, arrays in chunks.items()}


def issue_init_log(num_records: int, num_chunks: int, source: str) -> None:
    logger.info(
        "loading %s records in %d chunks from %s",
        long_num_format(num_records),
        num_chunks,
        source,
    )


def dataframe_to_numpy_dict(df: DataFrame) -> Tchunk:
    return {colnames: df[colnames].to_numpy() for colnames in df.columns}


class DataFrameReader(BaseReader):
    def __init__(
        self,
        data: DataFrame,
        *,
        columns: Iterable,
        chunksize: int | None = None,
        **reader_kwargs,
    ) -> None:
        super().__init__(data, columns=columns, chunksize=chunksize, **reader_kwargs)

    def _init_source(self, source: Any, **reader_kwargs) -> None:
        self._data = source

        super()._init_source(source)
        issue_init_log(self.num_records, self.num_chunks, "memory")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    @property
    def num_records(self) -> int:
        return len(self._data)

    def _load_next_chunk(self) -> Tchunk:
        start = self._chunk_idx * self.chunksize
        end = start + self.chunksize
        chunk = self._data[start:end]
        return dataframe_to_numpy_dict(chunk[self.columns])

    def read(self, sparse: int) -> Tchunk:
        return dataframe_to_numpy_dict(self._data[self.columns][::sparse])


class FileReader(BaseReader):
    def __init__(
        self,
        path: Tpath,
        *,
        columns: Iterable,
        chunksize: int | None = None,
        **reader_kwargs,
    ) -> None:
        super().__init__(path, columns=columns, chunksize=chunksize, **reader_kwargs)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._file.close()


class ParquetFile(Iterator):
    __slots__ = ("path", "_file", "columns", "_group_idx")

    def __init__(self, path: Tpath, columns: Iterable[str]) -> None:
        self.columns = list(columns)
        self.path = Path(path)
        self._file = parquet.ParquetFile(self.path)

        self.rewind()

    def close(self) -> None:
        self._file.close()

    @property
    def num_groups(self) -> int:
        return self._file.num_row_groups

    @property
    def num_records(self) -> int:
        return self._file.metadata.num_rows

    def rewind(self) -> None:
        self._group_idx = 0

    def __next__(self) -> Table:
        if self._group_idx >= self.num_groups:
            raise StopIteration

        group = self._file.read_row_group(self._group_idx, self.columns)
        self._group_idx += 1
        return group

    def __iter__(self) -> Iterator[Table]:
        self.rewind()
        return self

    def get_empty_group(self) -> Table:
        full_schema = self._file.schema.to_arrow_schema()
        schema = pa.schema([full_schema.field(name) for name in self.columns])
        return Table.from_pylist([], schema=schema)


class ParquetReader(FileReader):
    def _init_source(self, path: Tpath) -> None:
        self._file = ParquetFile(path, self.columns)
        self._cache = self._file.get_empty_group()

        super()._init_source(path)
        issue_init_log(
            self.num_records, self.num_chunks, f"from Parquet file: {self.path}"
        )

    @property
    def path(self) -> Path:
        return self._file.path

    @property
    def num_records(self) -> int:
        return self._file.num_records

    def _load_next_chunk(self) -> Tchunk:
        reached_end = False
        while len(self._cache) < self.chunksize:
            try:
                next_group = next(self._file)
                self._cache = pa.concat_tables([self._cache, next_group])
            except StopIteration:
                reached_end = True
                break

        if not reached_end:
            table = self._cache[: self.chunksize]
            self._cache = self._cache[self.chunksize :]
        else:
            table = self._cache

        return {colname: table.column(colname).to_numpy() for colname in self.columns}

    def __iter__(self) -> Iterator[Tchunk]:
        self._cache = self._file.get_empty_group()
        self._file.rewind()
        return super().__iter__()

    def read(self, sparse: int) -> Tchunk:
        return super().read(sparse)


class FitsReader(FileReader):
    def _init_source(self, source: Tpath, hdu: int = 1) -> None:
        self.path = Path(source)
        self._file = fits.open(str(self.path))
        self._hdu = self._file[hdu]

        super()._init_source(source)
        issue_init_log(
            self.num_records, self.num_chunks, f"from FITS file: {self.path}"
        )

    @property
    def num_records(self) -> int:
        return len(self._hdu.data)

    def _load_next_chunk(self) -> Tchunk:
        hdu_data = self._hdu.data
        offset = self._chunk_idx * self.chunksize
        group_slice = slice(offset, offset + self.chunksize)

        return {
            colname: swap_byteorder(hdu_data[colname][group_slice])
            for colname in self.columns
        }

    def read(self, sparse: int) -> Tchunk:
        return {
            colname: swap_byteorder(self._hdu.data[colname][::sparse])
            for colname in self.columns
        }


class HDFReader(FileReader):
    def _init_source(self, source: Tpath) -> None:
        self.path = Path(source)
        self._file = h5py.File(self.path, mode="r")

        super()._init_source(source)
        issue_init_log(
            self.num_records, self.num_chunks, f"from HDF5 file: {self.path}"
        )

    @property
    def num_records(self) -> int:
        num_records = [self._file[name].shape[0] for name in self.columns]
        if len(set(num_records)) != 1:
            raise IndexError("columns do not have equal length")
        return num_records[0]

    def _load_next_chunk(self) -> Tchunk:
        offset = self._chunk_idx * self.chunksize
        return {
            colname: self._file[colname][offset : offset + self.chunksize]
            for colname in self.columns
        }

    def read(self, sparse: int) -> Tchunk:
        return {colname: self._file[colname][::sparse] for colname in self.columns}


def new_filereader(
    path: Tpath,
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
    _, ext = os.path.splitext(str(path))
    ext = ext.lower()

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
