from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Protocol

import fitsio
import h5py
import numpy as np
import polars as pl
from polars import DataFrame
from pyarrow import parquet

from yaw.catalog.patch import PatchData, PatchDataCached
from yaw.catalog.utils import DataChunk, patch_path_from_id
from yaw.core.utils import TypePathStr

from ._streaming import _count_lines, _estimate_lines

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "count_lines",
    "estimate_lines",
    "ParquetReader",
    "FitsReader",
    "HDFReader",
    "CSVReader",
    "PatchCollector",
    "PatchWriter",
    "get_reader",
]


def count_lines(filename: str) -> int:
    return _count_lines(filename)


def estimate_lines(filename: str) -> int:
    return _estimate_lines(filename)


class Closable(Protocol):
    def close(self) -> None:
        pass


class FileContext(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> FileContext:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()


class Reader(FileContext):
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

    def _chunk_from_dataframe(self, data: DataFrame) -> DataChunk:
        data_dict = {col: data[col].to_numpy() for col in data.columns}
        return self._chunk_from_dict(data_dict)

    @abstractmethod
    def _init_file(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        self.file.close()

    @property
    @abstractmethod
    def n_rows(self) -> int:
        pass

    def estimate_nrows(self) -> int:
        return self.n_rows

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


class ParquetReader(Reader):
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
        # reading a sparse sample not directly supported by polars.read_parquet()
        if sparse is not None:
            return super().read_all(sparse)
        else:
            dataframe = pl.read_parquet(str(self.path), columns=self._colnames)
            return self._chunk_from_dataframe(dataframe)


class FitsReader(Reader):
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

    def _init_file(self, chunksize: int = 1_000_000, hdu: int = 1) -> None:
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


class HDFReader(Reader):
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

    def _init_file(self, chunksize: int = 1_000_000) -> None:
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


class CSVReader(Reader):
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
        batchsize: int = 10_000,
        separator: str = ",",
        eol_char: str = "\n",
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
            batchsize=batchsize,
            separator=separator,
            eol_char=eol_char,
        )

    def _init_file(
        self,
        chunksize: int = 1_000_000,
        batchsize: int = 10_000,
        separator: str = ",",
        eol_char: str = "\n",
    ) -> None:
        self.chunksize = chunksize
        self.batchsize = batchsize
        self.separator = separator
        self.eol_char = eol_char

    def close(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return count_lines(str(self.path))

    def estimate_nrows(self) -> int:
        return estimate_lines(str(self.path))

    def iter(self) -> Generator[DataChunk]:
        n_batch = self.chunksize // self.batchsize
        reader = pl.read_csv_batched(
            str(self.path),
            columns=self._colnames,
            separator=self.separator,
            eol_char=self.eol_char,
            batch_size=self.batchsize,
        )
        while True:
            batches = reader.next_batches(n_batch)
            if batches is None:
                return
            dataframe = pl.concat(batches)
            yield self._chunk_from_dataframe(dataframe)

    def read_all(self, sparse: int | None = None) -> DataChunk:
        # reading a sparse sample not directly supported by polars.read_csv()
        if sparse is not None:
            return super().read_all(sparse)
        else:
            dataframe = pl.read_csv(
                str(self.path),
                columns=self._colnames,
                separator=self.separator,
                eol_char=self.eol_char,
            )
            return self._chunk_from_dataframe(dataframe)


class Collector(FileContext):
    @abstractmethod
    def process(
        self,
        df: DataFrame,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class PatchCollector(Collector):
    def __init__(self) -> None:
        self._chunks: dict[int, list[DataChunk]] = defaultdict(list)

    def process(self, data: DataChunk) -> None:
        for pid, patch_chunk in data.groupby():
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

    def process(self, data: DataChunk) -> None:
        for pid, patch_chunk in data.groupby():
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
    if ext in (".csv",):
        reader = CSVReader
    elif ext in (".fits", ".cat"):
        reader = FitsReader
    elif ext in (".hdf5", ".hdf", ".h5"):
        reader = HDFReader
    elif ext in (".pqt", ".parq", ".parquet"):
        reader = ParquetReader
    else:
        raise ValueError(f"unrecognized file extesion '{ext}'")
    return reader
