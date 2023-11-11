from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generator, Iterable, Iterator

import fitsio
import h5py
import numpy as np
import pyarrow
from pyarrow import parquet
from pyarrow.interchange import from_dataframe

try:
    import polars as pl

    use_polars = True
except ImportError:
    import pandas as pd

    use_polars = False

if use_polars:
    DataFrame = pl.DataFrame
else:
    DataFrame = pd.DataFrame

if TYPE_CHECKING:
    from pyarrow import Table


def pyarrow_groupby(
    table: Table, by: str, rename: dict[str, str] | None = None
) -> Generator[tuple[Any, Table]]:
    col_names = table.column_names
    col_names.remove(by)
    if rename:
        out_names = [rename.get(col_name, col_name) for col_name in col_names]
    else:
        out_names = col_names
    aggregates = [(col_name, "list") for col_name in col_names]
    lists = table.group_by(by).aggregate(aggregates)

    for i, key in enumerate(lists.column(by)):
        group_col_data = [
            lists.column(f"{col_name}_list")[i].values for col_name in col_names
        ]
        group_table = pyarrow.Table.from_arrays(group_col_data, names=out_names)
        yield key, group_table


def dataframe_groupby(df: DataFrame, *args, **kwargs) -> Iterator[tuple[Any, DataFrame]]:
    if hasattr(df, "group_by"):
        return df.group_by(*args, **kwargs)
    else:
        return df.groupby(*args, **kwargs)


class FileContext(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> FileContext:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()


class Reader(FileContext):
    @abstractmethod
    def __init__(self, path: str, columns: Iterable[str] | None, **kwargs) -> None:
        pass

    def close(self) -> None:
        self.file.close()

    @abstractmethod
    def iter(self) -> Generator[DataFrame]:
        pass


class ParquetReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str] | None = None,
    ) -> None:
        self.path = path
        self.columns = None if columns is None else list(columns)
        self.file = parquet.ParquetFile(path)

    def iter(self) -> Generator[DataFrame]:
        n_groups = self.file.num_row_groups
        for gid in range(n_groups):
            group = self.file.read_row_group(gid, self.columns)
            if use_polars:
                yield pl.from_arrow(group)
            else:
                yield group.to_pandas()


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
        self.hdu = hdu
        self.chunksize = chunksize
        self.file = fitsio.FITS(path)
        if columns is None:
            self.columns = self.fits.get_colnames()
        else:
            self.columns = list(columns)

    def iter(self) -> Generator[DataFrame]:
        hdu = self.file[self.hdu]
        n_rows = hdu.get_nrows()
        n_groups = int(np.ceil(n_rows / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            data = hdu[self.columns][offset : offset + self.chunksize]
            coldata = {
                col: data[col].byteswap().newbyteorder() for col in data.dtype.fields
            }
            if use_polars:
                yield pl.DataFrame(coldata)
            else:
                yield pd.DataFrame(coldata)


class HDFReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str],
        *,
        chunksize: int = 1_000_000,
    ) -> None:
        self.path = path
        self.columns = list(columns)
        self.chunksize = chunksize
        self.file = h5py.File(path, mode="r")

    def iter(self) -> Generator[DataFrame]:
        n_rows = [self.file[col].shape[0] for col in self.columns]
        if len(set(n_rows)) != 1:
            raise IndexError("columns do not have equal length")
        n_groups = int(np.ceil(n_rows[0] / self.chunksize))
        for gid in range(n_groups):
            offset = gid * self.chunksize
            coldata = {
                col: self.file[col][offset : offset + self.chunksize]
                for col in self.columns
            }
            if use_polars:
                yield pl.DataFrame(coldata)
            else:
                yield pd.DataFrame(coldata)


class CSVReader(Reader):
    def __init__(
        self,
        path: str,
        columns: Iterable[str] | None = None,
        *,
        chunksize: int = 1_000_000,
        separator: str = ",",
        eol_char: str = "\n",
    ) -> None:
        self.path = path
        self.columns = None if columns is None else list(columns)
        self.chunksize = chunksize
        self.separator = separator
        self.eol_char = eol_char

    def close(self) -> None:
        pass

    def iter(self) -> Generator[DataFrame]:
        if use_polars:
            n_batch = 10
            batch_size = int(np.ceil(self.chunksize // n_batch))
            reader = pl.read_csv_batched(
                self.path,
                columns=self.columns,
                separator=self.separator,
                eol_char=self.eol_char,
                batch_size=batch_size,
            )
            while True:
                group = reader.next_batches(n_batch)
                if group is None:
                    return
                yield pl.concat(group)
        else:
            reader = pd.read_csv(
                self.path,
                usecols=self.columns,
                sep=self.separator,
                lineterminator=self.eol_char,
                chunksize=self.chunksize,
            )
            for batch in reader:
                yield batch


class PatchCollector:

    def __init__(self) -> None:
        self._patches: dict[int, list[Table | DataFrame]] = {}

    def process(
        self,
        df: DataFrame,
        patch_key: str,
        drop_key: bool = True,
    ) -> None:
        for pid, df_patch in dataframe_groupby(df, patch_key):
            if drop_key:
                if use_polars:
                    df_patch = df_patch.drop(patch_key)
                else:
                    df_patch = df_patch.drop(columns=patch_key)
            if pid not in self._patches:
                self._patches[pid] = []
            self._patches[pid].append(df_patch)

    def get_patches(self) -> dict[int, DataFrame]:
        patches = dict()
        for pid, shards in self._patches.items():
            if use_polars:
                patches[pid] = pl.concat(shards)
            else:
                patches[pid] = pd.concat(shards)
        self._patches = {}
        return patches

    def get_patches2(self) -> dict[int, DataFrame]:
        patches = dict()
        for pid, shards in self._patches.items():
            patch = pyarrow.concat_tables(shards)
            if use_polars:
                yield pl.from_arrow(patch)
            else:
                yield patch.to_pandas()
        self._patches = {}
        return patches


class PatchWriter(FileContext):
    def __init__(self, prefix: str) -> None:
        self.files: dict[int, pyarrow.OSFile] = {}
        self.writers: dict[int, parquet.ipc.new_file] = {}
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
        for pid, df_patch in dataframe_groupby(df, patch_key):
            if isinstance(df_patch, DataFrame):
                if use_polars:
                    if drop_key:
                        df_patch = df_patch.drop(patch_key)
                    arrow_patch = df_patch.to_arrow()
                else:
                    if drop_key:
                        df_patch = df_patch.drop(columns=patch_key)
                    arrow_patch = from_dataframe(df_patch)
            else:
                raise TypeError("'df' is not a valid DataFrame")

            if pid not in self.writers:
                self.files[pid] = pyarrow.OSFile(self.template.format(pid), "wb")
                self.writers[pid] = pyarrow.ipc.new_file(
                    self.files[pid],
                    arrow_patch.schema,
                )
            self.writers[pid].write(arrow_patch)
