from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from yaw import CorrData, CorrFunc, HistData, RedshiftData

if TYPE_CHECKING:
    from typing import Any


TypeDataFile = TypeVar("TypeDataFile", bound="DataFile")


class DataFile(ABC):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def exists(self) -> bool:
        return self.path.exists()

    @abstractmethod
    def load(self) -> Any:
        pass


class CorrDataFile(DataFile):
    def load(self) -> CorrData:
        template = self.path.with_suffix("")
        return CorrData.from_files(template)


class CorrFuncFile(DataFile):
    def load(self) -> CorrFunc:
        return CorrFunc.from_file(self.path)


class HistDataFile(DataFile):
    def load(self) -> HistData:
        template = self.path.with_suffix("")
        return HistData.from_files(template)


class RedshiftDataFile(DataFile):
    def load(self) -> RedshiftData:
        template = self.path.with_suffix("")
        return RedshiftData.from_files(template)


class TomographicDirectory(Mapping[int, TypeDataFile]):
    def __init__(self, file_type: TypeDataFile, directory: Path, template: str) -> None:
        template = str(template)
        if "?" not in template:
            raise ValueError(
                "path template must contain '?' as placeholder for the bin index"
            )
        self.template = template

        self.file_type = file_type
        self.directory = Path(directory)

    def get_path(self, idx: int) -> Path:
        return self.directory / self.template.replace("?", str(idx))

    def _get_re_template(self) -> str:
        pattern = re.escape(self.template).replace(r"\?", r"(\d+)")
        if not re.search(r"\.[a-zA-Z0-9]+$", pattern):
            pattern += r".*"

    def _idx_from_path(self, path: Path) -> Path:
        match = re.match(self._get_re_template(), str(path))
        if not match:
            raise ValueError(f"does not match '{self.template}': {path}")
        return int(match.group(1))

    def _glob(self) -> Iterator[Path]:
        template = self.template
        if self._get_re_template().endswith(".*"):
            template += ".*"
        yield from glob(self.directory / template)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __getitem__(self, idx: int) -> TypeDataFile:
        path = self._path_from_idx(idx)
        if not path.exists():
            raise FileNotFoundError(path)
        return self.file_type(path)

    def __iter__(self) -> Iterator[int]:
        yield from set(self._idx_from_path(path) for path in self._glob())


class ProjectDirectory:
    counts_dir = "paircounts"
    autocorr_ref = CorrFuncFile(f"{counts_dir}/auto_reference.hdf")
    autocorr_unk = TomographicDirectory(CorrFuncFile, counts_dir, "auto_unknown_?.hdf")
    crosscorr = TomographicDirectory(CorrFuncFile, counts_dir, "cross_?.hdf")

    estimate_dir = "estimate"
    w_ss = CorrDataFile(f"{estimate_dir}/auto_reference")
    w_pp = TomographicDirectory(CorrDataFile, estimate_dir, "auto_unknown_?")
    w_sp = TomographicDirectory(RedshiftDataFile, estimate_dir, "nz_cc_?")

    true_dir = "true"
    ref = HistDataFile(f"{true_dir}/nz_reference")
    unk = TomographicDirectory(HistDataFile, true_dir, "nz_true_?")
