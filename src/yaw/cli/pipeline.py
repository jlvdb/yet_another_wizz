from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yaw.cli.config import ProjectConfig


class Pipeline:
    progress: bool
    config: ProjectConfig

    def run(self) -> None:
        self.config.tasks.get_tasks()
        while len(self.config.task) > 0:
            task = self.tasks.pop()
            task.execute()
