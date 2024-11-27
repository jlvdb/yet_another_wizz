from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from yaw.cli.tasks import Task

if TYPE_CHECKING:
    from typing_extensions import Self

    from yaw.cli.config import ProjectConfig


class LockFile:
    def __init__(self, path: Path | str, content: str) -> None:
        self.path = Path(path)
        self.content = content

    def inspect(self) -> str | None:
        if not self.path.exists():
            return None

        with self.path.open() as f:
            return f.read()

    def acquire(self) -> None:
        if self.path.exists():
            raise FileExistsError(f"lock file already exists: {self.path}")

        with self.path.open(mode="w") as f:
            f.write(self.content)

    def release(self) -> None:
        self.path.unlink()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.release()


class Pipeline:
    progress: bool
    config: ProjectConfig

    def __init__(self, tasks):
        tasks_ = []
        for name, compl in tasks:
            task = Task.get(name)(...)
            task._completed = compl
            tasks_.append(task)
        self.tasks: set[Task] = set(tasks_)
        self._connect_inputs()

    def _connect_inputs(self) -> None:
        for task in self.tasks:
            for input_task in self.tasks:
                if input_task == task:
                    continue
                task.connect_input(input_task)

            task.check_inputs()

    def drop_cache(self) -> None: ...
