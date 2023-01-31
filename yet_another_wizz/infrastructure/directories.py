from __future__ import annotations

import os
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from pathlib import Path, _posix_flavour, _windows_flavour
from typing import Any


def _get_numeric_suffix(path: Path) -> int:
    base = path.with_suffix(suffix="").name
    _, num = base.rsplit("_", 1)
    return int(num)


class Directory(ABC, Path):
    _flavour = _windows_flavour if os.name == 'nt' else _posix_flavour

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


class CountsDirectory(Directory):

    _cross_prefix = "cross"
    _auto_prefix = "auto_unknown"

    def get_auto_reference(self) -> Path:
        return Path(self.joinpath("auto_reference.hdf"))

    def get_auto(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._auto_prefix}_{bin_idx}.hdf"))

    def get_cross(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._cross_prefix}_{bin_idx}.hdf"))


class EstimateDirectory(Directory):

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
