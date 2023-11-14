from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, Iterable, Protocol

import fitsio
import h5py
import numpy as np
import polars as pl
import pyarrow
from polars import DataFrame
from pyarrow import parquet

from yaw.catalogs.patches import PatchMeta

from ._streaming import _count_lines, _estimate_lines

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyarrow.ipc import RecordBatchFileWriter


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

    @abstractmethod
    def __init__(self, path: str, columns: Iterable[str] | None, **kwargs) -> None:
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
    def iter(self) -> Generator[DataFrame]:
        pass

    @abstractmethod
    def read_all(self, sparse: int | None = None) -> DataFrame:
        total = 0
        chunks = []
        for chunk in self.iter():
            n_chunk = len(chunk)
            if sparse is not None:
                # take a regular subset that factors in the chunk size
                idx = np.arange(total % sparse, n_chunk, sparse)
                chunk = chunk.select([pl.all().take(idx)])
            chunks.append(chunk)
            total += n_chunk
        return pl.concat(chunks)


class ParquetReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str] | None = None,
    ) -> None:
        self.path = path
        self.file = parquet.ParquetFile(path)
        if columns is None:
            self.columns = None
        else:
            availble = set(self.file.metadata.schema.names)
            self.columns = []
            for col in columns:
                if col not in availble:
                    raise KeyError(f"column '{col}' not found")
                self.columns.append(col)

    @property
    def n_rows(self) -> int:
        metadata = parquet.read_metadata(self.path)
        return metadata.num_rows

    def iter(self) -> Generator[DataFrame]:
        n_groups = self.file.num_row_groups
        for gid in range(n_groups):
            group = self.file.read_row_group(gid, self.columns)
            yield pl.from_arrow(group)

    def read_all(self, sparse: int | None = None) -> DataFrame:
        # reading a sparse sample not directly supported by polars.read_parquet()
        if sparse is not None:
            return super().read_all(sparse)
        else:
            return pl.read_parquet(self.path, columns=self.columns)


class FitsReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str] | None = None,
        *,
        chunksize: int = 1_000_000,
        hdu: int = 1,
    ) -> None:
        self.path = path
        self.chunksize = chunksize
        self.file = fitsio.FITS(path)
        self.hdu = self.file[hdu]
        if columns is None:
            self.columns = self.fits.get_colnames()
        else:
            self.columns = list(columns)

    @property
    def n_rows(self) -> int:
        return self.hdu.get_nrows()

    def iter(self) -> Generator[DataFrame]:
        n_groups = int(np.ceil(self.n_rows / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            data: NDArray = self.hdu[self.columns][offset : offset + self.chunksize]
            coldata = {
                col: data[col].byteswap().newbyteorder() for col in data.dtype.fields
            }
            yield DataFrame(coldata)

    def read_all(self, sparse: int | None = None) -> DataFrame:
        data: NDArray = self.hdu[self.columns][::sparse]
        coldata = {
            col: data[col].byteswap().newbyteorder() for col in data.dtype.fields
        }
        return DataFrame(coldata)


class HDFReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str],
        *,
        chunksize: int = 1_000_000,
    ) -> None:
        self.path = path
        if columns is None:
            raise ValueError(
                "columns (data set paths) must be specified for HDF5 files"
            )
        self.columns = list(columns)
        self.chunksize = chunksize
        self.file = h5py.File(path, mode="r")

    @property
    def n_rows(self) -> int:
        n_rows = [self.file[col].shape[0] for col in self.columns]
        if len(set(n_rows)) != 1:
            raise IndexError("columns do not have equal length")
        return n_rows[0]

    def iter(self) -> Generator[DataFrame]:
        n_groups = int(np.ceil(self.n_rows / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            coldata = {
                col: self.file[col][offset : offset + self.chunksize]
                for col in self.columns
            }
            yield DataFrame(coldata)

    def read_all(self, sparse: int | None = None) -> DataFrame:
        coldata = {col: self.file[col][::sparse] for col in self.columns}
        return DataFrame(coldata)


class CSVReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str] | None = None,
        *,
        chunksize: int = 1_000_000,
        batchsize: int = 10_000,
        separator: str = ",",
        eol_char: str = "\n",
    ) -> None:
        self.path = path
        self.columns = None if columns is None else list(columns)
        self.chunksize = chunksize
        self.batchsize = batchsize
        self.separator = separator
        self.eol_char = eol_char

    def close(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return count_lines(self.path)

    def estimate_nrows(self) -> int:
        return estimate_lines(self.path)

    def iter(self) -> Generator[DataFrame]:
        n_batch = self.chunksize // self.batchsize
        reader = pl.read_csv_batched(
            self.path,
            columns=self.columns,
            separator=self.separator,
            eol_char=self.eol_char,
            batch_size=self.batchsize,
        )
        while True:
            batches = reader.next_batches(n_batch)
            if batches is None:
                return
            yield pl.concat(batches)

    def read_all(self, sparse: int | None = None) -> DataFrame:
        # reading a sparse sample not directly supported by polars.read_csv()
        if sparse is not None:
            return super().read_all(sparse)
        else:
            return pl.read_csv(
                self.path,
                columns=self.columns,
                separator=self.separator,
                eol_char=self.eol_char,
            )


class Collector(ABC):
    @abstractmethod
    def process(
        self,
        df: DataFrame,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        pass


class PatchCollector(Collector):
    def __init__(self) -> None:
        self.metadata: dict[int, PatchMeta] = {}
        self.patches: dict[int, list[DataFrame]] = {}

    def process(
        self,
        df: DataFrame,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        for pid, df_patch in df.group_by(patch_key):
            if drop_key:
                df_patch = df_patch.drop(patch_key)
            if pid not in self.patches:
                self.metadata[pid] = PatchMeta.uninitialised(
                    has_w="weight" in df, has_z="redshift" in df
                )
                self.patches[pid] = []
            self.metadata[pid].accumulate(df_patch)
            self.patches[pid].append(df_patch)

    def get_data(self) -> dict[int, DataFrame]:
        patches = dict()
        for pid, shards in self.patches.items():
            patches[pid] = pl.concat(shards)
        return patches


class PatchWriter(FileContext, Collector):
    def __init__(self, prefix: str) -> None:
        self.metadata: dict[int, PatchMeta] = {}
        self.paths: dict[int, str] = {}
        self.files: dict[int, pyarrow.OSFile] = {}
        self.writers: dict[int, RecordBatchFileWriter] = {}
        root = os.path.dirname(prefix)
        if root != "" and not os.path.exists(root):
            os.mkdir(root)
        self.template = prefix + "_{:d}.feather"

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        for f in self.files.values():
            f.close()

    def process(
        self,
        df: DataFrame,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        for pid, df_patch in df.group_by(patch_key):
            if drop_key:
                df_patch = df_patch.drop(patch_key)
            arrow_patch = df_patch.to_arrow()
            if pid not in self.writers:
                self.metadata[pid] = PatchMeta.uninitialised(
                    has_w="weight" in df, has_z="redshift" in df
                )
                self.paths[pid] = self.template.format(pid)
                self.files[pid] = pyarrow.OSFile(self.paths[pid], "wb")
                self.writers[pid] = pyarrow.ipc.new_file(
                    self.files[pid],
                    arrow_patch.schema,
                )
            self.metadata[pid].accumulate(df_patch)
            self.writers[pid].write(arrow_patch)


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


def init_reader(
    path: str, columns: Iterable[str], reader: type[Reader] | None = None
) -> Reader:
    if reader is None:
        reader = get_reader(path)
    return reader(path, columns)
