from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from yet_another_wizz.core.utils import scales_to_keys


class CacheDirectory(Path):

    def get_reference(self) -> tuple(Path, Path):
        return (self.joinpath("reference"), self.joinpath("ref_rand"))

    def get_unknown(self, bin_idx: int) -> tuple(Path):
        return (
            self.joinpath(f"unknown_{bin_idx}"),
            self.joinpath(f"unk_rand_{bin_idx}"))

    def get_bin_indices(self) -> list[int]:
        idx = []
        for path in self.iterdir():
            if path.name.startswith("unknown"):
                _, num = path.name.split("_")
                idx.append(int(num))
        return sorted(idx)

    def iter_bins(self) -> Iterator[tuple(Path, Path)]:
        for idx in self.get_bin_indices():
            yield self.get_unknown(idx)


class CountsDirectory(Path):

    def get_auto_reference(self) -> Path:
        return self.joinpath("auto_reference.hdf")

    def get_cross(self, bin_idx: int) -> Path:
        return self.joinpath(f"cross_{bin_idx}.hdf")

    def get_auto_unknown(self, bin_idx: int) -> Path:
        return self.joinpath(f"auto_unknown_{bin_idx}.hdf")

    def get_bin_indices(self) -> list[int]:
        idx = []
        drop_suffix = ""
        for path in self.iterdir():
            if path.name.startswith("cross"):
                base = path.with_suffix(drop_suffix).name
                _, num = base.split("_")
                idx.append(int(num))
        return sorted(idx)

    def iter_bins(self) -> Iterator[tuple(Path, Path)]:
        for idx in self.get_bin_indices():
            yield self.get_cross(idx), self.get_auto_unknown(idx)


class EstimateDirectory(Path):
    pass


class ProjectDirectory(Path):

    @property
    def binning_file(self) -> Path:
        return self.path.joinpath("binning.dat")

    @property
    def config_file(self) -> Path:
        return self.path.joinpath("config.yaml")

    @property
    def inputs_file(self) -> Path:
        return self.path.joinpath("inputs.yaml")

    @property
    def cache_dir(self) -> Path:
        return self.path.joinpath()

    def list_caches(self) -> list(str):
        return [path.name for path in self.cache_dir.iterdir()]

    def get_cache(self, backend: str) -> CacheDirectory:
        path = self.cache_dir.joinpath(backend)
        if not path.exists():
            raise ValueError(f"cache for backend '{backend}' does not exist")
        return CacheDirectory(path)

    @property
    def counts_dir(self) -> Path:
        return self.path.joinpath()

    def list_counts_scales(self) -> list(str):
        return [path.name for path in self.counts_dir.iterdir()]

    def get_counts(self, scale_key: str) -> CacheDirectory:
        path = self.cache_dir.joinpath(scale_key)
        if not path.exists():
            raise ValueError(f"counts for scale '{scale_key}' do not exist")
        return CacheDirectory(path)

    @property
    def estimate_dir(self) -> EstimateDirectory:
        return self.path.joinpath("estimate")

    def list_estimate_scales(self) -> list(str):
        return [path.name for path in self.estimate_dir.iterdir()]

    def get_estimate(self, scale_key: str) -> EstimateDirectory:
        path = self.cache_dir.joinpath(scale_key)
        if not path.exists():
            raise ValueError(f"estimate for scale '{scale_key}' does not exist")
        return EstimateDirectory(path)
