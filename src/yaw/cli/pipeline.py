from __future__ import annotations

import itertools
from collections import Counter, deque
from typing import TYPE_CHECKING

from yaw.cli.tasks import Task

if TYPE_CHECKING:
    from collections.abc import Iterable

    from yaw.cli.config import ProjectConfig


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


def has_child(task: Task, candidates: set[Task]) -> bool:
    for candidate in candidates:
        if task in candidate.inputs | candidate.optionals:
            return True
    return False


def find_chain_ends(tasks: set[Task]) -> set[Task]:
    return {task for task in tasks if not has_child(task, tasks)}


def build_chain(end: Task, chain: deque[Task] | None = None) -> deque[Task]:
    chain = chain or deque((end,))

    for parent in end.inputs | end.optionals:
        if parent in chain:
            chain.remove(parent)
        chain.appendleft(parent)
        chain = build_chain(parent, chain)

    return chain


def remove_duplicates(tasks: Iterable[Task]) -> deque[Task]:
    seen = set()
    uniques = deque()
    for task in tasks:
        if task not in seen:
            uniques.append(task)
            seen.add(task)
    return uniques


def build_queue(tasks: set[Task], incl_complete: bool = False) -> deque[Task]:
    chains = [build_chain(end) for end in find_chain_ends(tasks)]
    queue = remove_duplicates(itertools.chain(*chains))
    if incl_complete:
        return queue
    return deque(task for task in queue if not task.completed())


def format_queue(queue: deque[Task]) -> str:
    tasks = Counter(task.name for task in queue)
    return " -> ".join(
        name if count == 1 else f"{name}[{count}]" for name, count in tasks.items()
    )


if __name__ == "__main__":
    proj = Pipeline(
        [
            ("true", False),
            ("cross", True),
            ("autoref", True),
            ("estimate", False),
            ("cacheref", False),
            ("cacheunk", False),
            ("estimate", False),
            ("estimate", False),
            ("plot", False),
        ],
    )

    print(format_queue(build_queue(proj.tasks, incl_complete=True)))
    print(format_queue(build_queue(proj.tasks)))
