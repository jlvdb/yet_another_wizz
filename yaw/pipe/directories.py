from __future__ import annotations

import os
import shutil
import textwrap
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from pathlib import Path, _posix_flavour, _windows_flavour
from typing import Any

from yaw.core.utils import bytes_format


def _get_numeric_suffix(path: Path) -> int:
    base = path.with_suffix(suffix="").name
    _, num = base.rsplit("_", 1)
    return int(num)


class Directory(Path):
    # seems to be the easiest way to subclass pathlib.Path
    _flavour = _windows_flavour if os.name == 'nt' else _posix_flavour


class CacheDirectory(Directory):

    def _generate_filenames(self, base: str) -> dict[str, Path]:
        tags = ("data", "rand")
        return {tag: Path(self.joinpath(f"{base}.{tag}")) for tag in tags}

    def get_reference(self) -> dict[str, Path]:
        return self._generate_filenames("reference")

    def get_unknown(self, bin_idx: int) -> dict[str, Path]:
        return self._generate_filenames(f"unknown_{bin_idx}")

    def get_bin_indices(self) -> set(int):
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith("unknown"))

    def summary(self) -> None:
        sizes = dict()
        for path in self.iterdir():
            if path.is_dir():
                sizes[path.name] = ("d", sum(
                    file.stat().st_size for file in path.rglob("*")))
            else:
                sizes[path.name] = ("f", path.stat().st_size)
        print(f"cache path: {self}")
        width = min(30, max(len(name) for name in sizes))
        for i, name in enumerate(sorted(sizes), 1):
            print_name = textwrap.shorten(name, width=width, placeholder="...")
            kind, bytes = sizes[name]
            bytes_fmt = bytes_format(bytes)
            print(f"{kind}> {print_name:{width}s}{bytes_fmt:>10s}")

    def drop(self, name: str) -> None:
        path = self.joinpath(name)
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()

    def drop_all(self) -> None:
        for path in self.iterdir():
            self.drop(path.name)


class DataDirectory(Directory, ABC):

    @abstractproperty
    def _cross_prefix(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def _auto_prefix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_auto_reference(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_auto(self, bin_idx: int) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_cross(self, bin_idx: int) -> Path:
        raise NotImplementedError

    def get_cross_indices(self) -> set[int]:
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith(self._cross_prefix))

    def get_auto_indices(self) -> set[int]:
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith(self._auto_prefix))

    def iter_cross(self) -> Iterator[tuple(int, Any)]:
        for idx in sorted(self.get_cross_indices()):
            yield idx, self.get_cross(idx)

    def iter_auto(self) -> Iterator[tuple(int, Any)]:
        for idx in sorted(self.get_auto_indices()):
            yield idx, self.get_auto(idx)


class CountsDirectory(DataDirectory):

    _cross_prefix = "cross"
    _auto_prefix = "auto_unknown"

    def get_auto_reference(self) -> Path:
        return Path(self.joinpath("auto_reference.hdf"))

    def get_auto(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._auto_prefix}_{bin_idx}.hdf"))

    def get_cross(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._cross_prefix}_{bin_idx}.hdf"))


class EstimateDirectory(DataDirectory):

    _cross_prefix = "nz_unknown"
    _auto_prefix = "auto_unknown"

    def _generate_filenames(self, base: str) -> dict[str, Path]:
        extensions = ("dat", "cov", "boot")
        return {ext: Path(self.joinpath(f"{base}.{ext}")) for ext in extensions}

    def get_auto_reference(self) -> dict[str, Path]:
        return self._generate_filenames("auto_reference")

    def get_auto(self, bin_idx: int) -> dict[str, Path]:
        return self._generate_filenames(f"{self._auto_prefix}_{bin_idx}")

    def get_cross(self, bin_idx: int) -> dict[str, Path]:
        return self._generate_filenames(f"{self._cross_prefix}_{bin_idx}")
