from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Union
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from pyarrow import parquet

from yaw.catalog.patch import DataChunk

__all__ = [
    "FitsReader",
    "HDFReader",
    "MemoryReader",
    "ParquetReader",
    "get_filereader",
]

TypePathStr = Union[Path, str]


def swap_byteorder(array: NDArray) -> NDArray:
    return array.view(array.dtype.newbyteorder()).byteswap()


class OptionalDependencyError(Exception):
    pass


class BaseReader(Iterator):
    _group_idx: int
    _num_groups: int

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name} @ {self._group_idx} / {self.num_chunks} chunks"

    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def __exit__(self, *args, **kwargs) -> None:
        pass

    @property
    @abstractmethod
    def num_records(self) -> int:
        pass

    @property
    def num_chunks(self) -> int:
        return self._num_groups

    @abstractmethod
    def _load_next_chunk(self) -> DataChunk:
        pass

    def __next__(self) -> DataChunk:
        if self._group_idx >= self._num_groups:
            raise StopIteration()
        chunk = self._load_next_chunk()
        self._group_idx += 1
        return chunk

    def __iter__(self) -> Iterator[DataChunk]:
        return self

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


class MemoryReader(BaseReader):
    def __init__(
        self,
        data: DataChunk,
        *,
        chunksize: int = 1_000_000,
        **kwargs,
    ) -> None:
        self._data = data
        self.chunksize = chunksize
        self._num_groups = int(np.ceil(self.num_records / self.chunksize))
        self._group_idx = 0

    def __exit__(self, *args, **kwargs) -> None:
        pass

    @property
    def num_records(self) -> int:
        return len(self._data)

    def _load_next_chunk(self) -> DataChunk:
        start = self._group_idx * self.chunksize
        end = start + self.chunksize
        return self._data[start:end]

    def read(self, sparse: int) -> DataChunk:
        return self._data[::sparse]


class FileReader(BaseReader):
    def __init__(
        self,
        path: TypePathStr,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int = 1_000_000,
        degrees: bool = True,
        **reader_kwargs,
    ) -> None:
        self.path = str(path)
        self._input_is_degrees = degrees
        self._init(ra_name, dec_name, weight_name, redshift_name, patch_name, chunksize)
        self._init_file(**reader_kwargs)

    def _init(
        self,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_name: str | None = None,
        chunksize: int = 1_000_000,
    ) -> None:
        attrs = ("ra", "dec", "weight", "redshift", "patch")  # DataChunk.from_columns
        columns = (ra_name, dec_name, weight_name, redshift_name, patch_name)
        self.attrs = tuple(attr for attr, col in zip(attrs, columns) if col is not None)
        self.columns = tuple(col for col in columns if col is not None)

        self.chunksize = chunksize
        self._group_idx = 0  # chunk iteration state

    def _convert_ra_dec(self, data: dict[str, NDArray]) -> None:
        if self._input_is_degrees:
            data["ra"] = np.deg2rad(data["ra"])
            data["dec"] = np.deg2rad(data["dec"])
        return data

    @abstractmethod
    def _init_file(self, **kwargs) -> None:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        self._file.close()


class ParquetReader(FileReader):
    def _init_file(self) -> None:
        self._file = parquet.ParquetFile(self.path)
        self._num_groups = self._file.num_row_groups

    @property
    def num_records(self) -> int:
        return self.file.metadata.num_rows

    def _load_next_chunk(self) -> DataChunk:
        group = self._file.read_row_group(self._group_idx, self.columns)
        data = {
            attr: group.column(col).to_numpy()
            for attr, col in zip(self.attrs, self.columns)
        }
        data = self._convert_ra_dec(data)
        return DataChunk.from_columns(**data)


class FitsReader(FileReader):
    def _init_file(self, hdu: int = 1) -> None:
        try:
            import fitsio
        except ImportError:
            raise OptionalDependencyError(
                "reading FITS files requires installing 'fitsio'"
            )

        self._file = fitsio.FITS(self.path)
        self._hdu = self._file[hdu]
        self._num_groups = int(np.ceil(self.num_records / self.chunksize))

    @property
    def num_records(self) -> int:
        return self._hdu.get_nrows()

    def _load_next_chunk(self) -> DataChunk:
        offset = self._group_idx * self.chunksize
        group = self._hdu[self.columns][offset : offset + self.chunksize]
        data = {
            attr: swap_byteorder(group[col])
            for attr, col in zip(self.attrs, self.columns)
        }
        data = self._convert_ra_dec(data)
        return DataChunk.from_columns(**data)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: swap_byteorder(self._hdu[col][::sparse])
            for attr, col in zip(self.attrs, self.columns)
        }
        data = self._convert_ra_dec(data)
        return DataChunk.from_columns(**data)


class HDFReader(FileReader):
    def _init_file(self) -> None:
        try:
            import h5py
        except ImportError:
            raise OptionalDependencyError(
                "reading HDF files requires installing 'h5py'"
            )

        self._file = h5py.File(self.path, mode="r")
        self._num_groups = int(np.ceil(self.num_records / self.chunksize))

    @property
    def num_records(self) -> int:
        num_records = [self._file[name].shape[0] for name in self.columns]
        if len(set(num_records)) != 1:
            raise IndexError("columns do not have equal length")
        return num_records[0]

    def _load_next_chunk(self) -> DataChunk:
        offset = self._group_idx * self.chunksize
        data = {
            attr: self._file[col][offset : offset + self.chunksize]
            for attr, col in zip(self.attrs, self.columns)
        }
        data = self._convert_ra_dec(data)
        return DataChunk.from_columns(**data)

    def read(self, sparse: int) -> DataChunk:
        data = {
            attr: self._file[col][::sparse]
            for attr, col in zip(self.attrs, self.columns)
        }
        data = self._convert_ra_dec(data)
        return DataChunk.from_columns(**data)


def get_filereader(path: TypePathStr) -> type[FileReader]:
    # parse the extension
    _, ext = os.path.splitext(str(path))
    ext = ext.lower()
    # get the correct reader
    if ext in (".fits", ".cat"):
        reader = FitsReader
    elif ext in (".hdf5", ".hdf", ".h5"):
        reader = HDFReader
    elif ext in (".pq", ".pqt", ".parq", ".parquet"):
        reader = ParquetReader
    else:
        raise ValueError(f"unrecognized file extesion '{ext}'")
    return reader
