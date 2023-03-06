from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


class TaskError(Exception):
    pass


class UndefinedTaskError(TaskError):

    def __init__(self, task: str) -> None:
        msg = f"got undefined task with name '{task}', options are: "
        msg += ", ".join(f"'{name}'" for name in Tasks)
        super().__init__(msg)


class TaskArgumentError(TaskError):

    def __init__(self, argument, taskname, options=None) -> None:
        msg = f"encountered unknown argument '{argument}' in task '{taskname}'"
        if options is not None:
            if len(options) == 0:
                msg += " (none allowed)"
            else:
                msg += " " + ", ".join(f"'{opt}'" for opt in options)
        super().__init__(msg)


@dataclass
class TaskRecord:

    name: str
    args: dict[str, Any] = field(default_factory=lambda: dict())

    def __eq__(self, other: TaskRecord) -> bool:
        if not isinstance(other, TaskRecord):
            raise TypeError(
                f"'==' not supported between instances of "
                f"'{type(self)}' and '{type(other)}'")
        return self.name == other.name

    @classmethod
    def restore(cls, data: dict[str, Any] | str) -> TaskRecord:
        if isinstance(data, str):
            return cls(data, dict())
        else:
            if len(data) != 1:
                raise TypeError(f"expected {dict} with single item or {str}")
            name, args = next(iter(data.items()))
            if not isinstance(args, dict):
                raise TypeError(
                    f"args must be of type {dict}, got {type(args)}")
            return cls(name, args)

    def save(self) -> dict[str, Any] | str:
        if len(self.args) == 0:
            return self.name
        else:
            return {self.name: {
                k: v for k, v in self.args.items() if v is not None}}


class Registry(Mapping):

    def __init__(self) -> None:
        super().__init__()
        self._register = {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._register})"

    def __getitem__(self, name: str) -> Callable:
        try:
            return self._register[name]
        except KeyError as e:
            raise KeyError(f"no item named '{name}' registered") from e

    def __iter__(self) -> Iterator[str]:
        return iter(self._register)

    def __len__(self) -> int:
        return len(self._register)

    def register(self, obj):
        self._register[obj.__name__] = obj
        return obj


class _Tasks(Registry):

    def __init__(self) -> None:
        super().__init__()
        self._order = {}

    def __iter__(self) -> Iterator[str]:
        for name in sorted(self._order, key=lambda name: self._order[name]):
            yield name

    def get_priority(self, task: TaskRecord | str) -> int:
        if isinstance(task, TaskRecord):
            task = task.name
        try:
            return self._order[task]
        except KeyError:
            raise UndefinedTaskError(task)

    def register(self, priority: int):
        def task(func):
            @wraps(func)
            def wrapper(args, *posargs, **kwargs) -> TaskRecord:
                from yaw.pipeline.project import ProjectDirectory

                with ProjectDirectory(args.wdir) as project:
                    setup_args = func(args, project, *posargs, **kwargs)
                    task = TaskRecord(func.__name__, setup_args)
                    project.add_task(task)
                return task
            name = wrapper.__name__
            self._register[name] = wrapper
            self._order[name] = priority
            return wrapper
        return task

Tasks = _Tasks()


class TaskList(Sequence):

    def __init__(self) -> None:
        self._tasks: list[TaskRecord] = []

    def __getitem__(self, item: int) -> TaskRecord:
        return self._tasks[item]

    def __len__(self) -> int:
        return len(self._tasks)

    def add(self, task: TaskRecord) -> None:
        if not isinstance(task, TaskRecord):
            raise TypeError(
                f"'task' must be of type {TaskRecord}, got {type(task)}")
        try:  # task exists -> overwrite
            idx = self._tasks.index(task)
            self._tasks[idx] = task
        except ValueError:  # insert maintaining order
            bisect.insort(
                self._tasks, task, key=lambda t: Tasks.get_priority(t))

    @classmethod
    def from_list(cls, task_list: Sequence[dict[str, Any] | str]) -> TaskList:
        new = cls()
        for task in task_list:
            task = TaskRecord.restore(task)
            new.add(task)
        return new

    def to_list(self) -> list[dict[str, Any] | str]:
        return [task.save() for task in self._tasks]
