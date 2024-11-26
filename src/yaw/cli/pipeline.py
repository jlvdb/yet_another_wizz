from __future__ import annotations

import itertools
from collections import deque
from typing import TYPE_CHECKING

from yaw.cli.tasks import Task

if TYPE_CHECKING:
    from collections.abc import Iterable

    from yaw.cli.config import ProjectConfig


class Pipeline:
    progress: bool
    config: ProjectConfig

    def __init__(self, tasks):
        self.tasks: list[Task] = set(Task.get(name)(...) for name in tasks)
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


def build_queue(tasks: set[Task]) -> list[deque[Task]]:
    chains = [build_chain(end) for end in find_chain_ends(tasks)]
    return remove_duplicates(itertools.chain(*chains))


if __name__ == "__main__":
    from timeit import default_timer

    proj = Pipeline(["true", "cross", "autoref", "cacheref", "cacheunk", "estimate"])

    N = 10000
    start = default_timer()
    for _ in range(N):
        queue = build_queue(proj.tasks)
    duration = default_timer() - start
    print(f"built queue in {duration / N * 1e6:.3f} μs")
    print()

    print(" -> ".join(task.name for task in queue))
