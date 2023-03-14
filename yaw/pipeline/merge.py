from __future__ import annotations

from pathlib import Path
from typing import Any

from collections.abc import Sequence

from yaw.utils import TypePathStr

from yaw.pipeline.project import YawDirectory
from yaw.pipeline.tasks import Task, TaskEstimateCorr, TaskManager, TaskPlot



MERGE_OPTIONS = ("bins", "patches", "redshift")


def merge_along_bins(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError


def merge_along_patches(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError


def merge_along_redshifts(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError


class MergedTaskManager(TaskManager):

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
        if mode not in MERGE_OPTIONS:
            raise ValueError(f"invalid merge mode '{mode}'")
        elif mode == "bins":
            merge_along_bins(paths)
        elif mode == "redshift":
            merge_along_redshifts(paths)
        else:
            merge_along_patches(paths)

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("merged.yaml")

    def setup_reload(self, setup: dict) -> None:
        super().setup_reload(setup)
        self._sources = tuple(Path(fpath) for fpath in setup.pop("sources", []))

    @property
    def sources(self) -> tuple[Path]:
        return self._sources

    def to_dict(self) -> dict[str, Any]:
        setup = dict(
            sources=[str(fpath) for fpath in self._sources],
            tasks=self._tasks.history_to_list())
        return setup
