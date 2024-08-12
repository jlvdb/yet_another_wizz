from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pyarrow import parquet

from yaw.catalog.utils import DataChunk

__all__ = [
    "DataFrameReader",
    "FitsReader",
    "HDFReader",
    "ParquetReader",
    "new_filereader",
]

Tpath = Union[Path, str]

CHUNKSIZE = 1_048_576


def swap_byteorder(array: NDArray) -> NDArray:
    return array.view(array.dtype.newbyteorder()).byteswap()


class OptionalDependencyError(Exception):
    pass


class BaseReader(Iterator[DataChunk], AbstractContextManager):
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
        chunksize: int = CHUNKSIZE,
        degrees: bool = True,
        **reader_kwargs,
    ) -> None:
        self.degrees = degrees
        self._init(ra_name, dec_name, weight_name, redshift_name, patch_name, chunksize)
        self._init_source(source, **reader_kwargs)

    def _init(
        self,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int = CHUNKSIZE,
    ) -> None:
        attrs = ("ra", "dec", "weights", "redshifts", "patch_ids")
        columns = (ra_name, dec_name, weight_name, redshift_name, patch_name)
        self.attrs = tuple(attr for attr, col in zip(attrs, columns) if col is not None)
        self.columns = tuple(col for col in columns if col is not None)

        self.chunksize = chunksize
        self._chunk_idx = 0

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
    def _load_next_chunk(self) -> DataChunk:
        pass

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
        chunksize: int = CHUNKSIZE,
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
        return super().read(sparse)


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
        chunksize: int = CHUNKSIZE,
        degrees: bool = True,
        **reader_kwargs,
    ) -> None:
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._file.close()


class ParquetReader(FileReader):
    def _init_source(self, path: Tpath) -> None:
        self.path = Path(path)
        self._file = parquet.ParquetFile(self.path)
        self._num_chunks = self._file.num_row_groups

    @property
    def num_records(self) -> int:
        return self._file.metadata.num_rows

    def _load_next_chunk(self) -> DataChunk:
        group = self._file.read_row_group(self._chunk_idx, self.columns)
        data = {
            attr: group.column(col).to_numpy()
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def read(self, sparse: int) -> DataChunk:
        return super().read(sparse)


class FitsReader(FileReader):
    def _init_source(self, source: Tpath, hdu: int = 1) -> None:
        try:
            import fitsio
        except ImportError:
            raise OptionalDependencyError(
                "reading FITS files requires installing 'fitsio'"
            )

        self.path = Path(source)
        self._file = fitsio.FITS(self.path)
        self._hdu = self._file[hdu]
        super()._init_source(source)

    @property
    def num_records(self) -> int:
        return self._hdu.get_nrows()

    def _load_next_chunk(self) -> DataChunk:
        offset = self._chunk_idx * self.chunksize
        group = self._hdu[self.columns][offset : offset + self.chunksize]

        data = {
            attr: swap_byteorder(group[col])
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: swap_byteorder(self._hdu[col][::sparse])
            for attr, col in zip(self.attrs, self.columns)
        }
        return DataChunk.from_columns(**data, degrees=self.degrees, chkfinite=True)


class HDFReader(FileReader):
    def _init_source(self, source: Tpath) -> None:
        try:
            import h5py
        except ImportError:
            raise OptionalDependencyError(
                "reading HDF files requires installing 'h5py'"
            )

        self.path = Path(source)
        self._file = h5py.File(self.path, mode="r")
        super()._init_source(source)

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
    chunksize: int = CHUNKSIZE,
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
