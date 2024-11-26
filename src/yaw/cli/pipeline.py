from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yaw.cli.config import ProjectConfig


class Pipeline:
    progress: bool
    config: ProjectConfig

    def __init__(self, tasks):
        self.tasks = list(tasks)
        self._connect_inputs()

    def _connect_inputs(self) -> None:
        for task in self.tasks:
            for input_task in self.tasks:
                if input_task == task:
                    continue
                task.connect_input(input_task)

            task.check_inputs()

    def run(self) -> None:
        self.config.tasks.get_tasks()
        while len(self.config.task) > 0:
            task = self.tasks.pop()
            task.execute()
