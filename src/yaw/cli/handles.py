from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from yaw import Catalog, CorrData, CorrFunc, HistData, RedshiftData
from yaw.catalog.catalog import PATCH_INFO_FILE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

T = TypeVar("T")


class Handle(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def load(self) -> Any:
        pass


class SingleFileHandle(Handle):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def exists(self) -> bool:
        return self.path.exists()


class CatalogHandle(SingleFileHandle):
    def exists(self) -> bool:
        return (self.path / PATCH_INFO_FILE).exists()

    def load(self) -> Catalog:
        return Catalog(self.path)


class CorrFuncHandle(SingleFileHandle):
    def load(self) -> CorrFunc:
        return CorrFunc.from_file(self.path)


class MultiFileHandle(Handle):
    def __init__(self, template: Path | str) -> None:
        self.template = Path(template)
        if self.template.suffix:
            raise ValueError("multi-file templates must not have any file extension")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.template}.*)"

    def exists(self) -> bool:
        for _ in self.template.glob():
            return True
        return False


class CorrDataHandle(MultiFileHandle):
    def load(self) -> CorrData:
        return CorrData.from_files(self.template)


class RedshiftDataHandle(MultiFileHandle):
    def load(self) -> RedshiftData:
        return RedshiftData.from_files(self.template)


class HistDataHandle(MultiFileHandle):
    def load(self) -> HistData:
        return HistData.from_files(self.template)


class CacheHandle(Handle):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def exists(self) -> bool:
        return CacheHandle(self.path / "data").exists()

    def load(self) -> tuple[Catalog, Catalog | None]:
        data = CacheHandle(self.path / "data").load()

        if CacheHandle(self.path / "rand").exists():
            rand = CacheHandle(self.path / "rand").load()
        else:
            rand = None

        return data, rand


class TomographyWrapper(Mapping[int, T]):
    def __init__(
        self, handle_type: type[T], template: Path | str, indices: Iterable[int]
    ) -> None:
        if not issubclass(handle_type, Handle):
            raise TypeError("'handle_type' must be a subclass of 'Handle'")
        self.type = handle_type

        self.template = str(template)
        if "?" not in self.template:
            raise ValueError("'template' must contain '?' as placeholder for indices")

        self._handles: dict[int, T]
        for idx in indices:
            source = self.template.replace("?", str(idx))
            self.handles[idx] = self.type(source)

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
        return all(handle.exists() for handle in self.handles.values())
