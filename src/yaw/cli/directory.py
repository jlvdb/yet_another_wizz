from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

from yaw.cli.handles import (
    CacheHandle,
    CorrDataHandle,
    CorrFuncHandle,
    HistDataHandle,
    RedshiftDataHandle,
    TomographyWrapper,
)
from yaw.cli.setup import Setup

if TYPE_CHECKING:
    from collections.abc import Iterable


class Directory:
    def __init__(self, path: Path | str, bin_indices: Iterable[int]) -> None:
        self.indices = list(bin_indices)

        self.path = Path(path)
        self.path.mkdir(exist_ok=True)


class CacheDirectory(Directory):
    @property
    def reference(self) -> CacheHandle:
        return CacheHandle(self.path / "reference")

    @property
    def unknown(self) -> TomographyWrapper[CacheHandle]:
        return TomographyWrapper(CacheHandle, self.path / "unknown_?", self.indices)


class PaircountsDirectory(Directory):
    @property
    def cross(self) -> TomographyWrapper[CorrFuncHandle]:
        return TomographyWrapper(
            CorrFuncHandle, self.path / "cross_?.hdf", self.indices
        )

    @property
    def auto_ref(self) -> CorrFuncHandle:
        return CorrFuncHandle(self.path / "auto_ref.hdf")

    @property
    def auto_unk(self) -> TomographyWrapper[CorrFuncHandle]:
        return TomographyWrapper(
            CorrFuncHandle, self.path / "auto_unk_?.hdf", self.indices
        )


class EstimateDirectory(Directory):
    @property
    def nz_est(self) -> TomographyWrapper[RedshiftDataHandle]:
        return TomographyWrapper(
            RedshiftDataHandle, self.path / "nz_est_?", self.indices
        )

    @property
    def auto_ref(self) -> CorrDataHandle:
        return CorrDataHandle(self.path / "auto_ref")

    @property
    def auto_unk(self) -> TomographyWrapper[CorrDataHandle]:
        return TomographyWrapper(CorrDataHandle, self.path / "auto_unk_?", self.indices)


class TrueDirectory(Directory):
    @property
    def reference(self) -> HistDataHandle:
        return HistDataHandle(self.path / "reference")

    @property
    def unknown(self) -> TomographyWrapper[HistDataHandle]:
        return TomographyWrapper(HistDataHandle, self.path / "nz_true_?", self.indices)


class ProjectDirectory:
    setup: Setup

    def __init__(
        self,
        path: Path | str,
        bin_indices: Iterable[int],
        *,
        overwrite: bool = False,
    ) -> None:
        self.indices = list(bin_indices)

        self.path = Path(path)
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(f"project directory exists: {self.path}")
            elif overwrite and not self.setup_path.exists():
                raise FileExistsError(
                    f"can only overwrite valid cache directories: {self.path}"
                )
            rmtree(self.path)
        self.path.mkdir(exist_ok=True)

    @property
    def setup_path(self) -> Path:
        return self.path / "setup.yml"

    @property
    def cache(self) -> CacheDirectory:
        return CacheDirectory(self.path / "cache")

    @property
    def paircounts(self) -> PaircountsDirectory:
        return PaircountsDirectory(self.path / "paircounts")

    @property
    def estimate(self) -> EstimateDirectory:
        return EstimateDirectory(self.path / "estimate")

    @property
    def true(self) -> TrueDirectory:
        return TrueDirectory(self.path / "true")
