from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import numpy as np

from yaw import AngularCoordinates
from yaw.cli.handles import (
    CacheHandle,
    CorrDataHandle,
    CorrFuncHandle,
    HistDataHandle,
    RedshiftDataHandle,
    TomographyWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class Directory:
    def __init__(self, path: Path | str, bin_indices: Iterable[int]) -> None:
        self.indices = list(bin_indices)

        self.path = Path(path)
        self.path.mkdir(exist_ok=True)


class CacheDirectory(Directory):
    @property
    def patch_center_file(self) -> Path:
        return self.path / "patch_centers.npy"

    @property
    def reference(self) -> CacheHandle:
        return CacheHandle(self.path / "reference")

    @property
    def unknown(self) -> TomographyWrapper[CacheHandle]:
        return TomographyWrapper(CacheHandle, self.path / "unknown_?", self.indices)

    def get_patch_centers(self) -> AngularCoordinates | None:
        if not self.patch_center_file.exists():
            return None
        data = np.load(self.patch_center_file)
        return AngularCoordinates(data)

    def set_patch_centers(self, centers: AngularCoordinates) -> None:
        if self.patch_center_file.exists():
            raise RuntimeError("overwriting existing patch centers not permitted")
        with self.patch_center_file.open(mode="wb") as f:
            np.save(f, centers.data)


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


class PlotDirectory(Directory):
    @property
    def auto_ref_path(self) -> Path:
        return self.path / "auto_ref.png"

    @property
    def auto_unk_path(self) -> Path:
        return self.path / "auto_unk.png"

    @property
    def redshifts_path(self) -> Path:
        return self.path / "redshifts.png"


class ProjectDirectory:
    def __init__(
        self,
        path: Path | str,
        bin_indices: Iterable[int],
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"project directory does not exist: {self.path}")
        if not self.indicator_path.exists():
            raise FileNotFoundError(f"not a valie project directory: {self.path}")

        self.indices = list(bin_indices)

    @classmethod
    def create(
        cls,
        path: Path | str,
        bin_indices: Iterable[int],
        *,
        overwrite: bool = False,
    ) -> None:
        new = cls.__new__(cls)  # need cls.indicator_path
        new.path = Path(path)

        if new.path.exists():
            if not overwrite:
                raise FileExistsError(f"project directory exists: {path}")
            elif not new.indicator_path.exists():
                raise FileExistsError(
                    f"can only overwrite valid cache directories: {path}"
                )
            rmtree(path)
        new.path.mkdir()

        with open(new.indicator_path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return cls(path, bin_indices)

    @property
    def indicator_path(self) -> Path:
        return self.path / ".project_info"

    @property
    def config_path(self) -> Path:
        return self.path / "pipeline.yml"

    @property
    def log_path(self) -> Path:
        return self.path / "pipeline.log"

    @property
    def lock_path(self) -> Path:
        return self.path / ".tasklock"

    @property
    def cache(self) -> CacheDirectory:
        return CacheDirectory(self.path / "cache", self.indices)

    @property
    def paircounts(self) -> PaircountsDirectory:
        return PaircountsDirectory(self.path / "paircounts", self.indices)

    @property
    def estimate(self) -> EstimateDirectory:
        return EstimateDirectory(self.path / "estimate", self.indices)

    @property
    def true(self) -> TrueDirectory:
        return TrueDirectory(self.path / "true", self.indices)

    @property
    def plot(self) -> PlotDirectory:
        return PlotDirectory(self.path / "plots", self.indices)