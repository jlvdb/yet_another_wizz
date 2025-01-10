"""
Implements data handles which link data products to paths in a file system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from yaw import Catalog, CorrData, CorrFunc, HistData, RedshiftData
from yaw.catalog.catalog import PATCH_INFO_FILE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")


class Handle(ABC, Generic[T]):
    """Base class for all handles."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Whether the linked file exists."""
        pass

    @abstractmethod
    def load(self) -> T:
        """Load and restore the data instance from the linked file, or raise a
        `FileNotFoundError`."""
        pass


class SingleFileHandle(Handle[T]):
    """Base class for a handle for a single file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def exists(self) -> bool:
        return self.path.exists()


class CatalogHandle(SingleFileHandle[Catalog]):
    """
    Handle for a cached `~yaw.Catalog`.

    Args:
        path:
            Path to cache directory.
    """

    def exists(self) -> bool:
        return (self.path / PATCH_INFO_FILE).exists()

    def load(self) -> Catalog:
        return Catalog(self.path)


class CorrFuncHandle(SingleFileHandle[CorrFunc]):
    """
    Handle for a cached `~yaw.CorrFunc`.

    Args:
        path:
            Path to HDF5 file.
    """

    def load(self) -> CorrFunc:
        return CorrFunc.from_file(self.path)


class MultiFileHandle(Handle[T]):
    """
    Handle for homogeneous set of files with a common name but different file
    extension.

    Args:
        template:
            Common file path without file extension..
    """

    def __init__(self, template: Path | str) -> None:
        self.template = Path(template)
        if self.template.suffix:
            raise ValueError("multi-file templates must not have any file extension")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.template}.*)"

    def exists(self) -> bool:
        for _ in glob(f"{self.template}.*"):
            return True
        return False


class CorrDataHandle(MultiFileHandle[CorrData]):
    """
    Handle for `~yaw.CorrData` stored in a set of text files.

    Args:
        template:
            Template with wild-cards that identify all linked files, must not
            have any file extension included.
    """

    def load(self) -> CorrData:
        return CorrData.from_files(self.template)


class RedshiftDataHandle(MultiFileHandle[RedshiftData]):
    """
    Handle for `~yaw.RedshiftData` stored in a set of text files.

    Args:
        template:
            Template with wild-cards that identify all linked files, must not
            have any file extension included.
    """

    def load(self) -> RedshiftData:
        return RedshiftData.from_files(self.template)


class HistDataHandle(MultiFileHandle[HistData]):
    """
    Handle for `~yaw.HistData` stored in a set of text files.

    Args:
        template:
            Template with wild-cards that identify all linked files, must not
            have any file extension included.
    """

    def load(self) -> HistData:
        return HistData.from_files(self.template)


class CacheHandle(Handle[tuple[Catalog, Catalog | None]]):
    """
    Handle for a pair cached data and optional random catalog.

    The data catalogs are cached in subdirectories called ``data`` and ``rand``.

    Args:
        path:
            Path to directory that holds the cached catalogs.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

        self.data = CatalogHandle(self.path / "data")
        self.rand = CatalogHandle(self.path / "rand")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def exists(self) -> bool:
        return self.data.exists()

    def load_data(self) -> Catalog:
        """Restore the cached data catalog or raise a `FileNotFoundError`."""
        return self.data.load()

    def load_rand(self) -> Catalog | None:
        """Restore the optional random catalog or return ``None`` if it does not
        exists."""
        if not self.rand.exists():
            return None
        return self.rand.load()

    def load(self) -> tuple[Catalog, Catalog | None]:
        """Restore data and random catalogs and return them as a tuple, the
        random catalog my be ``None``."""
        return self.load_data(), self.load_rand()


class TomographyWrapper(Mapping[int, T]):
    """
    A wrapper for handles that simplifies dealing with data product that exist
    for each tomographic bin (e.g. redshift estimates).

    Handles are identified by a tomographic bin index. The wrapper object is
    subscriptable and iteratable, which yields the individual handles for the
    given bin index.

    Args:
        handle_type:
            The handle class to wrap.
        template:
            Path template for wrapped data files, must contain a single ``?``
            which will be replaced by the corresponding bin index.
        indices:
            A list of bin indices that this wrapper will handle.
    """

    def __init__(
        self, handle_type: type[T], template: Path | str, indices: Iterable[int]
    ) -> None:
        if not issubclass(handle_type, Handle):
            raise TypeError("'handle_type' must be a subclass of 'Handle'")
        self.type = handle_type

        self.template = str(template)
        if "?" not in self.template:
            raise ValueError("'template' must contain '?' as placeholder for indices")

        self._handles: dict[int, T] = {}
        for idx in indices:
            source = self.template.replace("?", str(idx))
            self._handles[idx] = self.type(source)

    def __repr__(self) -> str:
        handles = ", ".join(repr(handle) for handle in self.values())
        return f"{type(self).__name__}({handles})"

    def __len__(self) -> int:
        return len(self._handles)

    def __getitem__(self, idx: int) -> T:
        return self._handles[idx]

    def __iter__(self) -> Iterator[int]:
        yield from self._handles

    def exists(self) -> bool:
        """Checks whether all individual handles exist."""
        return all(handle.exists() for handle in self._handles.values())
