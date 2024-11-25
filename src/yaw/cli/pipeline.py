from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from yaw.cli.tasks import Task


class Pipeline:
    def __init__(self, tasks: Iterable[Task]):
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
        while len(self.tasks) > 0:
            task = self.tasks.pop()
            task.execute()


if __name__ == "__main__":
    from yaw.cli import tasks

    pipe = Pipeline(
        [
            tasks.CacheRefTask(),
            tasks.CacheUnkTask(),
            tasks.CrossTask(),
            tasks.AutoRefTask(),
            tasks.EstimateTask(),
            tasks.TrueTask(),
        ]
    )
    pipe.run()
