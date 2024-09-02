from __future__ import annotations

import logging
import os
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sized
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pyarrow as pa
from astropy.io import fits
from pyarrow import parquet

from yaw.catalog.utils import DataChunk

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from pyarrow import Table
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

CHUNKSIZE = 16_777_216

logger = logging.getLogger(__name__)


def long_num_format(x: float | int) -> str:
    """
    Format a floating point number as string with a numerical suffix.

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


@dataclass
class DummyReader(Sized, Iterable[None], AbstractContextManager):
    num_chunks: int
    has_weights: bool
    has_redshifts: bool

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __len__(self) -> int:
        return self.num_chunks

    def __iter__(self) -> Iterator[None]:
        for _ in range(self.num_chunks):
            yield None


class BaseReader(Sized, Iterator[DataChunk], AbstractContextManager):
    @abstractmethod
    def __init__(
        self,
        source: Any,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
        degrees: bool = True,
        **reader_kwargs,
    ) -> None:
        self.degrees = degrees

        self._init(ra_name, dec_name, weight_name, redshift_name, patch_name, chunksize)
        self._init_source(source, **reader_kwargs)

        logger.debug("selecting input columns: %s", ", ".join(self.columns))

    def _init(
        self,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int | None = None,
    ) -> None:
        attrs = ("ra", "dec", "weights", "redshifts", "patch_ids")
        columns = (ra_name, dec_name, weight_name, redshift_name, patch_name)
        self.attrs = tuple(attr for attr, col in zip(attrs, columns) if col is not None)
        self.columns = tuple(col for col in columns if col is not None)

        self.chunksize = chunksize or CHUNKSIZE
        self._chunk_idx = 0

    @abstractmethod
    def _init_source(self, source: Any, **reader_kwargs) -> None:
        self._num_chunks = int(np.ceil(self.num_records / self.chunksize))

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name} @ {self._chunk_idx} / {self.num_chunks} chunks"

    @property
    def has_redshifts(self) -> bool:
        return "redshifts" in self.attrs

    @property
    def has_weights(self) -> bool:
        return "weights" in self.attrs

    @property
    @abstractmethod
    def num_records(self) -> int:
        pass

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    def get_dummy(self) -> DummyReader:
        return DummyReader(self.num_chunks, self.has_weights, self.has_redshifts)

    @abstractmethod
    def _load_next_chunk(self) -> DataChunk:
        pass

    def __len__(self) -> int:
        return self.num_chunks

    def __next__(self) -> DataChunk:
        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        chunk = self._load_next_chunk()
        self._chunk_idx += 1
        return chunk

    def __iter__(self) -> Iterator[DataChunk]:
        self._chunk_idx = 0
        return self

    @abstractmethod
    def read(self, sparse: int) -> DataChunk:
        n_read = 0
        chunks = []
        for chunk in self:
            # keep track of where the next spare record in the new chunk is located
            chunk_offset = ((n_read // sparse + 1) * sparse - n_read) % sparse
            chunk_size = len(chunk)

            sparse_idx = np.arange(chunk_offset, chunk_size, sparse)
            chunks.append(chunk[sparse_idx])
            n_read += chunk_size

        return DataChunk.from_chunks(chunks)


def issue_io_log(num_records: int, num_chunks: int, source: str) -> None:
    logger.info(
        "loading %s records in %d chunks from %s",
        long_num_format(num_records),
        num_chunks,
        source,
    )


class DataFrameReader(BaseReader):
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
        **reader_kwargs,
    ) -> None:
        super().__init__(
            data,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
            **reader_kwargs,
        )

    def _init_source(self, source: Any, **reader_kwargs) -> None:
        self._data = source

        super()._init_source(source)
        issue_io_log(self.num_records, self.num_chunks, "memory")

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"iter_state={self._chunk_idx}/{self.num_chunks}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    @property
    def num_records(self) -> int:
        return len(self._data)

    def _load_next_chunk(self) -> DataChunk:
        start = self._chunk_idx * self.chunksize
        end = start + self.chunksize
        chunk = self._data[start:end]

        data = {
            attr: chunk[col].to_numpy() for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: self._data[col][::sparse].to_numpy()
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)


class FileReader(BaseReader):
    def __init__(
        self,
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
    ) -> None:
        self.path = Path(path)
        super().__init__(
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

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"iter_state={self._chunk_idx}/{self.num_chunks}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.path}"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._file.close()


class ParquetFile(Iterator):
    __slots__ = ("path", "_file", "columns", "_group_idx")

    def __init__(self, path: Tpath, columns: Iterable[str]) -> None:
        self.columns = tuple(columns)
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
        return pa.Table.from_pylist([], schema=schema)


class ParquetReader(FileReader):
    def _init_source(self, path: Tpath) -> None:
        self._file = ParquetFile(path, self.columns)
        self._cache = self._file.get_empty_group()

        super()._init_source(path)
        issue_io_log(
            self.num_records, self.num_chunks, f"from Parquet file: {self.path}"
        )

    @property
    def path(self) -> Path:
        return self._file.path

    @property
    def num_records(self) -> int:
        return self._file.num_records

    def _load_next_chunk(self) -> DataChunk:
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

        data = {
            attr: table.column(col).to_numpy()
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def __iter__(self) -> Iterator[DataChunk]:
        self._cache = self._file.get_empty_group()
        self._file.rewind()
        return super().__iter__()

    def read(self, sparse: int) -> DataChunk:
        return super().read(sparse)


def swap_byteorder(array: NDArray) -> NDArray:
    return array.view(array.dtype.newbyteorder()).byteswap()


class FitsReader(FileReader):
    def _init_source(self, path: Tpath, hdu: int = 1) -> None:
        self._file = fits.open(str(path))
        self._hdu = self._file[hdu]

        super()._init_source(path)
        issue_io_log(self.num_records, self.num_chunks, f"from FITS file: {self.path}")

    @property
    def num_records(self) -> int:
        return len(self._hdu.data)

    def _load_next_chunk(self) -> DataChunk:
        hdu_data = self._hdu.data
        offset = self._chunk_idx * self.chunksize
        group_slice = slice(offset, offset + self.chunksize)

        data = {
            attr: swap_byteorder(hdu_data[col][group_slice])
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: swap_byteorder(self._hdu.data[col][::sparse])
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)


class HDFReader(FileReader):
    def _init_source(self, path: Tpath) -> None:
        self._file = h5py.File(path, mode="r")

        super()._init_source(path)
        issue_io_log(self.num_records, self.num_chunks, f"from HDF5 file: {self.path}")

    @property
    def num_records(self) -> int:
        num_records = [self._file[name].shape[0] for name in self.columns]
        if len(set(num_records)) != 1:
            raise IndexError("columns do not have equal length")
        return num_records[0]

    def _load_next_chunk(self) -> DataChunk:
        offset = self._chunk_idx * self.chunksize
        data = {
            attr: self._file[col][offset : offset + self.chunksize]
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: self._file[col][::sparse]
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)


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
