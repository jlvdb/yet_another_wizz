from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, Protocol

import fitsio
import h5py
import numpy as np
from pyarrow import csv, parquet

from yaw.catalog.patch import PatchData, PatchDataCached
from yaw.catalog.utils import DataChunk, patch_path_from_id
from yaw.core.utils import TypePathStr

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "ParquetReader",
    "FitsReader",
    "HDFReader",
    "CsvReader",
    "PatchCollector",
    "PatchWriter",
    "get_reader",
]


class Closable(Protocol):
    def close(self) -> None:
        ...


class FileContext(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> FileContext:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()


class Reader(Protocol):
    def _init_file(self, **kwargs) -> None:
        ...

    def close(self) -> None:
        ...

    @property
    def n_rows(self) -> int:
        ...

    def iter(self) -> Generator[DataChunk]:
        ...

    def read_all(self, sparse: int | None = None) -> DataChunk:
        ...


class ChunkReader(Reader, FileContext):
    def __init__(self, data: DataChunk, degrees: bool = True, **kwargs) -> None:
        self.data = data
        self.degrees = degrees

    def _init_file(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return len(self.data)

    def iter(self) -> Generator[DataChunk]:
        yield self.data

    def read_all(self, sparse: int | None = None) -> DataChunk:
        if sparse is None:
            return self.data
        else:
            return self.data[::sparse]


class BaseReader(Reader, FileContext):
    file: Closable

    def __init__(
        self,
        path: TypePathStr,
        ra_name: str,
        dec_name: str,
        patch_name: str | None = None,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> None:
        self._degrees = degrees  # whether conversion to radian is needed
        # determine how to map columns
        self._col_to_chunk_arg = dict(ra=ra_name, dec=dec_name)
        if weight_name is not None:
            self._col_to_chunk_arg["weight"] = weight_name
        if redshift_name is not None:
            self._col_to_chunk_arg["redshift"] = redshift_name
        if patch_name is not None:
            self._col_to_chunk_arg["patch"] = patch_name
        # get the list of columns to read
        self._colnames = list(self._col_to_chunk_arg.values())
        # open the file, set up reader, etc.
        self.path = Path(path)
        self._init_file(**kwargs)

    def _chunk_from_dict(self, data: dict[str, NDArray]) -> DataChunk:
        converted = {new: data[old] for new, old in self._col_to_chunk_arg.items()}
        if self._degrees:
            converted["ra"] = np.deg2rad(converted["ra"])
            converted["dec"] = np.deg2rad(converted["dec"])
        return DataChunk(**converted)

    @abstractmethod
    def _init_file(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        self.file.close()

    @property
    @abstractmethod
    def n_rows(self) -> int:
        pass

    @abstractmethod
    def iter(self) -> Generator[DataChunk]:
        pass

    @abstractmethod
    def read_all(self, sparse: int | None = None) -> DataChunk:
        total = 0
        chunks = []
        for chunk in self.iter():
            n_chunk = len(chunk)
            if sparse is not None:
                # take a regular subset that factors in the chunk size
                idx = np.arange(total % sparse, n_chunk, sparse)
                chunk = chunk[idx]
            chunks.append(chunk)
            total += n_chunk
        return DataChunk.from_chunks(chunks)


class ParquetReader(BaseReader):
    file: parquet.ParquetFile

    def _init_file(self) -> None:
        self.file = parquet.ParquetFile(str(self.path))
        availble = set(self.file.metadata.schema.names)
        for col in self._colnames:
            if col not in availble:
                raise KeyError(f"column '{col}' not found")

    @property
    def n_rows(self) -> int:
        metadata = self.file.metadata
        return metadata.num_rows

    def iter(self) -> Generator[DataChunk]:
        n_groups = self.file.num_row_groups
        for gid in range(n_groups):
            group = self.file.read_row_group(gid, self._colnames)
            data = {column._name: column.to_numpy() for column in group.columns}
            yield self._chunk_from_dict(data)

    def read_all(self, sparse: int | None = None) -> DataChunk:
        if sparse is not None:
            return super().read_all(sparse)
        else:
            table = self.file.read(self._colnames)
            data = {column._name: column.to_numpy() for column in table.columns}
            return self._chunk_from_dict(data)


class FitsReader(BaseReader):
    file: fitsio.FITS

    def __init__(
        self,
        path: str,
        ra_name: str,
        dec_name: str,
        patch_name: str | None = None,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        chunksize: int = 1_000_000,
        hdu: int = 1,
    ) -> None:
        super().__init__(
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            degrees=degrees,
            # backend specific
            chunksize=chunksize,
            hdu=hdu,
        )

    def _init_file(self, chunksize: int, hdu: int) -> None:
        self.chunksize = chunksize
        self.file = fitsio.FITS(str(self.path))
        self.hdu = self.file[hdu]

    @property
    def n_rows(self) -> int:
        return self.hdu.get_nrows()

    def iter(self) -> Generator[DataChunk]:
        n_groups = int(np.ceil(self.n_rows / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            data: NDArray = self.hdu[self._colnames][offset : offset + self.chunksize]
            data = {
                col: data[col].byteswap().newbyteorder() for col in data.dtype.fields
            }
            yield self._chunk_from_dict(data)

    def read_all(self, sparse: int | None = None) -> DataChunk:
        data: NDArray = self.hdu[self._colnames][::sparse]
        data = {col: data[col].byteswap().newbyteorder() for col in data.dtype.fields}
        return self._chunk_from_dict(data)


class HDFReader(BaseReader):
    file: h5py.File

    def __init__(
        self,
        path: str,
        ra_name: str,
        dec_name: str,
        patch_name: str | None = None,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        chunksize: int = 1_000_000,
    ) -> None:
        super().__init__(
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            degrees=degrees,
            # backend specific
            chunksize=chunksize,
        )

    def _init_file(self, chunksize: int) -> None:
        self.chunksize = chunksize
        self.file = h5py.File(str(self.path), mode="r")

    @property
    def n_rows(self) -> int:
        n_rows = [self.file[col].shape[0] for col in self._colnames]
        if len(set(n_rows)) != 1:
            raise IndexError("columns do not have equal length")
        return n_rows[0]

    def iter(self) -> Generator[DataChunk]:
        n_groups = int(np.ceil(self.n_rows / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            data = {
                col: self.file[col][offset : offset + self.chunksize]
                for col in self._colnames
            }
            yield self._chunk_from_dict(data)

    def read_all(self, sparse: int | None = None) -> DataChunk:
        data = {col: self.file[col][::sparse] for col in self._colnames}
        return self._chunk_from_dict(data)


class CsvReader(BaseReader):
    file: csv.CSVStreamingReader

    def __init__(
        self,
        path: str,
        ra_name: str,
        dec_name: str,
        patch_name: str | None = None,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        chunksize: int = 1_000_000,
        skip_rows: int | None = None,
        delimiter: str | None = None,
        quote_char: str | Literal[False] = '"',
        escape_char: str | Literal[False] = False,
    ) -> None:
        super().__init__(
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            degrees=degrees,
            # backend specific
            chunksize=chunksize,
            skip_rows=skip_rows,
            delimiter=delimiter,
            quote_char=quote_char,
            escape_char=escape_char,
        )

    def _init_file(
        self,
        chunksize: int,
        skip_rows: int | None,
        delimiter: str | None,
        quote_char: str | Literal[False],
        escape_char: str | Literal[False],
    ) -> None:
        self.chunksize = chunksize
        self.read_options = csv.ReadOptions(
            use_threads=True,
            skip_rows=skip_rows,
        )
        self.parse_options = csv.ParseOptions(
            delimiter=delimiter,
            quote_char=quote_char,
            escape_char=escape_char,
        )
        self._reopen_file()  # run any checks that pyarrow might do

    def _reopen_file(self) -> csv.CSVStreamingReader:
        self.file = csv.open_csv(
            str(self.path),
            read_options=self.read_options,
            parse_options=self.parse_options,
        )

    @property
    def n_rows(self) -> int:
        # this is not very fast, but there is no other way and nobody should use
        # CSV for large numerical datasets in the first place in 2023
        with open(self.path) as file:
            return sum(1 for _ in file)

    def iter(self) -> Generator[DataChunk]:
        self._reopen_file()
        for group in self.file:
            data = {colname: group[colname].to_numpy() for colname in self._colnames}
            yield self._chunk_from_dict(data)

    def read_all(self, sparse: int | None = None) -> DataChunk:
        if sparse is not None:
            return super().read_all(sparse)
        else:
            self._reopen_file()
            table = self.file.read_all()
            data = {column._name: column.to_numpy() for column in table.columns}
            return self._chunk_from_dict(data)


class Collector(FileContext):
    @abstractmethod
    def process(
        self,
        chunk: DataChunk,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class PatchCollector(Collector):
    def __init__(self) -> None:
        self._chunks: dict[int, list[DataChunk]] = defaultdict(list)

    def process(self, chunk: DataChunk) -> None:
        for pid, patch_chunk in chunk.groupby():
            self._chunks[pid].append(patch_chunk)

    def get_patches(self) -> dict[int, PatchData]:
        data = {
            pid: DataChunk.from_chunks(chunks) for pid, chunks in self._chunks.items()
        }
        return {pid: PatchData(pid, **data.to_dict()) for pid, data in data.items()}


class PatchWriter(Collector):
    def __init__(self, cache_directory: TypePathStr) -> None:
        self._patches: dict[int, PatchDataCached] = dict()
        # set up cache directory
        self.cache_directory = Path(cache_directory)
        if not self.cache_directory.exists():
            self.cache_directory.mkdir(parents=True)

    def process(self, chunk: DataChunk) -> None:
        for pid, patch_chunk in chunk.groupby():
            if pid not in self._patches:
                cachepath = patch_path_from_id(self.cache_directory, pid)
                if os.path.exists(cachepath):
                    shutil.rmtree(cachepath)
                self._patches[pid] = PatchDataCached.empty(
                    cachepath,
                    pid,
                    has_weight=patch_chunk.weight is not None,
                    has_redshift=patch_chunk.redshift is not None,
                )
            self._patches[pid].append_data(**patch_chunk.to_dict())

    def get_patches(self) -> dict[int, PatchDataCached]:
        return self._patches


def get_reader(path: str) -> type[Reader]:
    # parse the extension
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    # get the correct reader
    if ext in (".csv"):
        reader = CsvReader
    elif ext in (".fits", ".cat"):
        reader = FitsReader
    elif ext in (".hdf5", ".hdf", ".h5"):
        reader = HDFReader
    elif ext in (".pqt", ".parq", ".parquet"):
        reader = ParquetReader
    else:
        raise ValueError(f"unrecognized file extesion '{ext}'")
    return reader
