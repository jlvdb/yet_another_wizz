from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, Protocol

import numpy as np
from pyarrow import csv, parquet

from yaw.catalog.utils import DataChunk

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pyarrow import RecordBatch, Table

    from yaw.core.utils import TypePathStr

__all__ = [
    "ParquetReader",
    "FitsReader",
    "HDFReader",
    "CsvReader",
    "get_reader",
]


def arrow_to_numpy_dict(arrow_data: RecordBatch | Table) -> dict[str, NDArray]:
    return {col_data._name: col_data.to_numpy() for col_data in arrow_data.columns}


class OptionalDependencyError(Exception):
    pass


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
    def __init__(
        self,
        data: DataChunk,
        degrees: bool = True,
        chunksize: int = 1_000_000,
        **kwargs,
    ) -> None:
        self.data = data
        self.degrees = degrees
        self.chunksize = chunksize

    def _init_file(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return len(self.data)

    def iter(self) -> Generator[DataChunk]:
        for offset in range(0, len(self.data), self.chunksize):
            yield self.data[offset : offset + self.chunksize]

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
        self._degrees = degrees

        self._colname_to_chunk_attr = dict(ra=ra_name, dec=dec_name)
        if weight_name is not None:
            self._colname_to_chunk_attr["weight"] = weight_name
        if redshift_name is not None:
            self._colname_to_chunk_attr["redshift"] = redshift_name
        if patch_name is not None:
            self._colname_to_chunk_attr["patch"] = patch_name

        self._colnames = list(self._colname_to_chunk_attr.values())
        self.path = Path(path)
        self._init_file(**kwargs)

    def _chunk_from_dict(self, data: dict[str, NDArray]) -> DataChunk:
        converted = {new: data[old] for new, old in self._colname_to_chunk_attr.items()}
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
        current_size = 0
        chunks = []
        for chunk in self.iter():
            chunk_size = len(chunk)
            if sparse is not None:
                chunk_offset = current_size % sparse
                sparse_idx = np.arange(chunk_offset, chunk_size, sparse)
                chunk = chunk[sparse_idx]
            chunks.append(chunk)
            current_size += chunk_size
        return DataChunk.from_chunks(chunks)


class ParquetReader(BaseReader):
    file: parquet.ParquetFile

    def _init_file(self) -> None:
        self.file = parquet.ParquetFile(str(self.path))
        availble_columns = set(self.file.metadata.schema.names)
        for column in self._colnames:
            if column not in availble_columns:
                raise KeyError(f"column '{column}' not found")

    @property
    def n_rows(self) -> int:
        metadata = self.file.metadata
        return metadata.num_rows

    def iter(self) -> Generator[DataChunk]:
        n_groups = self.file.num_row_groups
        for gid in range(n_groups):
            group = self.file.read_row_group(gid, self._colnames)
            yield self._chunk_from_dict(arrow_to_numpy_dict(group))

    def read_all(self, sparse: int | None = None) -> DataChunk:
        if sparse is not None:
            return super().read_all(sparse)
        else:
            table = self.file.read(self._colnames)
            return self._chunk_from_dict(arrow_to_numpy_dict(table))


class FitsReader(BaseReader):
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
        try:
            import fitsio
        except ImportError:
            raise OptionalDependencyError(
                "reading FITS files requires installing 'fitsio'"
            )
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
        data = {
            colname: data[colname].byteswap().newbyteorder()
            for colname in data.dtype.fields
        }
        return self._chunk_from_dict(data)


class HDFReader(BaseReader):
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
        try:
            import h5py
        except ImportError:
            raise OptionalDependencyError(
                "reading HDF files requires installing 'h5py'"
            )
        self.chunksize = chunksize
        self.file = h5py.File(str(self.path), mode="r")

    @property
    def n_rows(self) -> int:
        n_rows = [self.file[colname].shape[0] for colname in self._colnames]
        if len(set(n_rows)) != 1:
            raise IndexError("columns do not have equal length")
        return n_rows[0]

    def iter(self) -> Generator[DataChunk]:
        n_groups = int(np.ceil(self.n_rows / self.chunksize))
        for group_index in range(n_groups):
            offset = group_index * self.chunksize
            data = {
                colname: self.file[colname][offset : offset + self.chunksize]
                for colname in self._colnames
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
        self._reopen_file()

    def _reopen_file(self) -> csv.CSVStreamingReader:
        self.file = csv.open_csv(
            str(self.path),
            read_options=self.read_options,
            parse_options=self.parse_options,
        )

    @property
    def n_rows(self) -> int:
        # this is not very fast, but there is no other way and nobody should use
        # CSV for large numerical datasets in 2023 in the first place
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
            data = {colname: table[colname].to_numpy() for colname in self._colnames}
            return self._chunk_from_dict(data)


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
