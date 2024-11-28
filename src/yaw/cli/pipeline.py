from __future__ import annotations

import logging
import random
import string
from pathlib import Path
from shutil import rmtree

import yaml

from yaw._version import __version_tuple__
from yaw.cli.config import ProjectConfig
from yaw.cli.directory import ProjectDirectory
from yaw.cli.logging import init_file_logging
from yaw.cli.tasks import TaskList
from yaw.utils import write_yaml

logger = logging.getLogger("yaw.cli.pipeline")


class LockFile:
    def __init__(self, path: Path | str, content: str) -> None:
        self.path = Path(path)
        self.content = content

    def inspect(self) -> str | None:
        if not self.path.exists():
            return None

        with self.path.open() as f:
            return f.read()

    def acquire(self, *, resume: bool = False) -> None:
        if self.path.exists() and not resume:
            raise FileExistsError(f"lock file already exists: {self.path}")

        with self.path.open(mode="w") as f:
            f.write(self.content)

    def release(self) -> None:
        self.path.unlink()


def read_config(setup_file: Path | str) -> tuple[ProjectConfig, TaskList]:
    with open(setup_file) as f:
        config_dict = yaml.safe_load(f)
        task_list = config_dict.pop("tasks")

    config = ProjectConfig.from_dict(config_dict)
    tasks = TaskList.from_list(task_list)
    return config, tasks


def write_config(
    setup_file: Path | str, config: ProjectConfig, tasks: TaskList
) -> None:
    version = ".".join(str(v) for v in __version_tuple__[:3])
    header = f"yaw_cli v{version} configuration"

    config_dict = config.to_dict()
    config_dict["tasks"] = tasks.to_list()

    with open(setup_file, mode="w") as f:
        write_yaml(config_dict, f, header_lines=[header], indent=4)


class Pipeline:
    def __init__(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        tasks: TaskList,
        *,
        cache_path: Path | str | None = None,
        max_workers: int | None = None,
        resume: bool = False,
        progress: bool = True,
    ) -> None:
        logger.info(f"using project directory: {directory.path}")

        self.directory = directory
        self.config = config
        self._update_cache_path(cache_path)
        self._update_max_workers(max_workers)

        self.tasks = tasks
        self.tasks.schedule(self.directory, resume=resume)
        self._validate()

        self._resume = resume
        self.progress = progress

    @classmethod
    def create(
        cls,
        project_path: Path | str,
        setup_file: Path | str,
        *,
        cache_path: Path | str | None = None,
        max_workers: int | None = None,
        overwrite: bool = False,
        resume: bool = False,
        progress: bool = True,
    ) -> Pipeline:
        config, tasks = read_config(setup_file)
        bin_indices = config.get_bin_indices()

        if Path(project_path).exists() and not overwrite:
            directory = ProjectDirectory(project_path)
        else:
            directory = ProjectDirectory.create(
                project_path, bin_indices, overwrite=overwrite
            )

        init_file_logging(directory.log_path)
        logger.info(f"using configuration from: {setup_file}")

        write_config(directory.config_path, config, tasks)
        return cls(
            directory,
            config,
            tasks,
            resume=resume,
            progress=progress,
            cache_path=cache_path,
            max_workers=max_workers,
        )

    def _update_cache_path(self, root_path: Path | str | None) -> None:
        if root_path is None:
            return
        if self.directory.cache_exists():
            raise FileExistsError("cannot update existing cache directory")

        root_path = Path(root_path)
        cache_name = "yaw_cache_" + "".join(random.choices(string.hexdigits, k=8))
        cache_path = (root_path / cache_name).resolve()
        logger.info(f"using external cache directory: {cache_path}")

        self.directory.link_cache(cache_path)
        cache_path.mkdir(exist_ok=True)

    def _update_max_workers(self, max_workers: int | None) -> None:
        self.config.correlation = self.config.correlation.modify(
            max_workers=max_workers
        )

    def _validate(self) -> None:
        NotImplemented

    def run(self) -> None:
        if len(self.tasks) > 0:
            msg = "resuming" if self._resume else "running"
            msg = msg + f" tasks: {self.tasks}"
            logger.info(msg)
        else:
            logger.warning("nothing to do")

        while self.tasks:
            task = self.tasks.pop()
            logger.client(f"running '{task.name}'")

            lock = LockFile(self.directory.lock_path, task.name)
            try:
                lock.acquire(resume=self._resume)
            except FileExistsError:
                msg = (
                    "previous pipeline finished unexpectedly or another "
                    "pipeline is already runnig; run with 'resume' option"
                )
                raise RuntimeError(msg)

            task.run(self.directory, self.config, progress=self.progress)

            lock.release()

        logger.client("done")

    def drop_cache(self) -> None:
        logger.client("dropping cache directory")

        if not self.directory.cache.path.exists():
            return

        if self.directory.cache.path.is_symlink():
            rmtree(self.directory.cache.path.resolve())
            self.directory.cache.path.unlink()

        else:
            rmtree(self.directory.cache.path)
