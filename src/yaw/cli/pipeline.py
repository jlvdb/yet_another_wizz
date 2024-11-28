from __future__ import annotations

from pathlib import Path

import yaml

from yaw._version import __version_tuple__
from yaw.cli.config import ProjectConfig
from yaw.cli.directory import ProjectDirectory
from yaw.cli.tasks import TaskList
from yaw.cli.utils import print_message
from yaw.utils import write_yaml


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
        resume: bool = False,
        progress: bool = True,
    ) -> None:
        print_message(
            f"using project directory: {directory.path}", colored=False, bold=False
        )
        self.directory = directory
        self.config = config

        self.tasks = tasks
        self.tasks.schedule(self.directory, resume=resume)
        self._validate()

        self.resume = resume
        self.progress = progress

    @classmethod
    def create(
        cls,
        wdir: Path | str,
        setup_file: Path | str,
        *,
        overwrite: bool = False,
        resume: bool = False,
        progress: bool = True,
    ) -> Pipeline:
        print_message(f"reading configuration: {setup_file}", colored=True, bold=False)
        config, tasks = read_config(setup_file)
        indices = config.get_bin_indices()

        if Path(wdir).exists() and not overwrite:
            directory = ProjectDirectory(wdir)
        else:
            directory = ProjectDirectory.create(wdir, indices, overwrite=overwrite)

        write_config(directory.config_path, config, tasks)
        return cls(directory, config, tasks, resume=resume, progress=progress)

    def _validate(self) -> None:
        NotImplemented

    def run(self) -> None:
        if len(self.tasks) > 0:
            msg = "resuming" if self.resume else "running"
            msg = msg + f" tasks: {self.tasks}"
            print_message(msg, colored=False, bold=False)
        else:
            print_message("nothing to do", colored=False, bold=False)

        while self.tasks:
            task = self.tasks.pop()
            print_message(f"running '{task.name}'", colored=True, bold=True)

            lock = LockFile(self.directory.lock_path, task.name)
            try:
                lock.acquire(resume=self.resume)
            except FileExistsError:
                msg = (
                    "previous pipeline finished unexpectedly or another "
                    "pipeline is already runnig; run with 'resume' option"
                )
                raise RuntimeError(msg)

            task.run(self.directory, self.config, progress=self.progress)

            lock.release()

        print_message("done", colored=True, bold=True)

    def drop_cache(self) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    from yaw.utils import get_logger

    get_logger("debug")

    pipe = Pipeline.create("project", "project.yml", resume=True)
    pipe.run()
