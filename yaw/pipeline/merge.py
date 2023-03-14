from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from collections.abc import Sequence

from yaw.utils import TypePathStr

from yaw.pipeline.processing import DataProcessor
from yaw.pipeline.project import ProjectDirectory, YawDirectory
from yaw.pipeline.tasks import Task, TaskEstimateCorr, TaskManager, TaskPlot



MERGE_OPTIONS = ("bins", "patches", "redshift")



def _determine_scales(
    projects: Sequence[ProjectDirectory]
) -> tuple[set[str], set[str]]:
    scale_sets = []
    for project in projects:
        scale_sets.append(set(scale for scale, _ in project.iter_counts()))
    common = set()
    extra = set()
    for scales in scale_sets:
        common &= scales
        extra |= scales
    extra = extra - common
    return common, extra


def _determine_bins(
    projects: Sequence[ProjectDirectory]
) -> tuple[set[int], set[int]]:
    bin_sets = []
    for project in projects:
        bin_sets.append(project.get_bin_indices())
    common = set()
    extra = set()
    for bins in bin_sets:
        common &= bins
        extra |= bins
    extra = extra - common
    return common, extra


def merge_along_bins(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend
    raise NotImplementedError


def merge_along_patches(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend, rweight
    raise NotImplementedError


def merge_along_redshifts(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend, rweight
    raise NotImplementedError


_not_impl_msg = "operation not implemented on merged data"


class MergedProcessor(DataProcessor):

    def load_reference(self, skip_rand: bool = False) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def load_unknown(self, skip_rand: bool = False) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def compute_linkage(self) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def run_auto_ref(self, *, compute_rr: bool) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def run_auto_unk(self, *, compute_rr: bool) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def run_cross(self, *, compute_rr: bool) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def write_nz_ref(self) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def write_nz_true(self) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def write_total_unk(self) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)

    def drop_cache(self) -> NoReturn:
        raise NotImplementedError(_not_impl_msg)


class MergedManager(TaskManager):

    def _insert_task(self, task: Task, task_list: list[Task]) -> None:
        if not isinstance(task, (TaskEstimateCorr, TaskPlot)):
            raise NotImplementedError(
                f"task '{task.get_name()}' not supported after merging")
        return super()._insert_task(task, task_list)


class MergedDirectory(YawDirectory):

    @classmethod
    def from_projects(
        cls,
        paths: Sequence[TypePathStr],
        mode: str
    ) -> MergedDirectory:
        projects = []
        for path in paths:
            projects.append(ProjectDirectory(path))
        if mode not in MERGE_OPTIONS:
            raise ValueError(f"invalid merge mode '{mode}'")
        elif mode == "bins":
            merge_along_bins(projects)
        elif mode == "redshift":
            merge_along_redshifts(projects)
        else:
            merge_along_patches(projects)

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("merged.yaml")

    def setup_reload(self, setup: dict) -> None:
        super().setup_reload(setup)
        self._sources = tuple(Path(fpath) for fpath in setup.pop("sources", []))
        # set up task management
        task_list = setup.get("tasks", [])
        self._tasks = MergedManager.from_history_list(task_list, project=self)

    @property
    def sources(self) -> tuple[Path]:
        return self._sources

    def to_dict(self) -> dict[str, Any]:
        setup = dict(
            sources=[str(fpath) for fpath in self._sources],
            tasks=self._tasks.history_to_list())
        return setup

    def get_bin_indices(self) -> set[int]:
        for scale in self.iter_scales():
            counts = self.get_counts_dir(scale)
            return counts.get_cross_indices() | counts.get_auto_indices()

    @property
    def n_bins(self) -> int:
        return len(self.get_bin_indices())
