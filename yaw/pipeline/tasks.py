from __future__ import annotations

import bisect
import functools
import logging
from abc import abstractclassmethod, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields, asdict, MISSING
from typing import TYPE_CHECKING, Any

from yaw import default as DEFAULT
from yaw.config import ResamplingConfig
from yaw.estimators import CorrelationEstimator
from yaw.utils import DictRepresentation, Parameter

if TYPE_CHECKING:  # pragma: no cover
    from argparse import Namespace
    from yaw.pipeline.project import ProjectDirectory


logger = logging.getLogger(__name__)

ESTIMATORS = [est.short for est in CorrelationEstimator.variants]
METHOD_OPTIONS = ResamplingConfig.implemented_methods


class TaskError(Exception):
    pass


class UndefinedTaskError(TaskError):

    def __init__(self, task: str) -> None:
        msg = f"got undefined task with name '{task}', options are: "
        msg += ", ".join(f"'{name}'" for name in Task._tasks)
        super().__init__(msg)


class TaskArgumentError(TaskError):

    def __init__(self, argument, taskname, options=None) -> None:
        msg = f"encountered unknown argument '{argument}' in task '{taskname}'"
        msg += ", options are:"
        if options is not None:
            if len(options) == 0:
                msg += " no arguments"
            else:
                msg += " " + ", ".join(f"'{opt}'" for opt in options)
        super().__init__(msg)


@functools.total_ordering
@dataclass(frozen=True)
class Task(DictRepresentation):
    _tasks = {}
    _order = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.get_name()
        if name in cls._tasks:
            raise TaskError(f"task with name '{name}' already registered")
        if name != "task":  # skip any meta tasks
            cls._tasks[name] = cls
            cls._order[name] = len(cls._order)

    @classmethod
    def all_tasks(cls) -> tuple[Task]:
        return tuple(cls._tasks.values())

    def __post_init__(self) -> None:
        for par in fields(self):
            if par.default is MISSING:
                name = f"{self.__class__.__name__}.{par.name}"
                raise TaskError(f"no default value for '{name}' provided")

    def __eq__(self, other: Task) -> bool:
        return Task._order[self.get_name()] == Task._order[other.get_name()]

    def __le__(self, other: Task) -> bool:
        return Task._order[self.get_name()] < Task._order[other.get_name()]

    @classmethod
    def from_argparse(cls, args: Namespace) -> Task:
        kwargs = {}
        for name in cls.get_parnames():
            kwargs[name] = getattr(args, name)
        return cls.from_dict(kwargs)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> Task:
        accepted = cls.get_parnames()
        for argname in the_dict:
            if argname not in accepted:
                raise TaskArgumentError(argname, cls.get_name(), accepted)
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        # return only non-default values
        defaults = self.get_defaults()
        return {k: v for k, v in asdict(self).items() if v != defaults[k]}

    @classmethod
    def get_parnames(cls) -> list[str]:
        return [par.name for par in fields(cls)]

    @classmethod
    def get_defaults(cls) -> dict[str, Any]:
        return {par.name: par.default for par in fields(cls)}

    @abstractclassmethod
    def get_name(cls) -> str:
        return "task"

    @abstractclassmethod
    def get_help(cls) -> str:
        return "task"

    @abstractmethod
    def __call__(self, project: ProjectDirectory) -> Any:
        project.add_task(self)
        logger.info(f"running task '{self.get_name()}'")
        args = ", ".join(f"{k}={repr(v)}" for k, v in asdict(self).items())
        if len(args) == 0:
            args = "---"
        logger.debug(f"arguments: {args}")


class RepeatableTask(Task):

    @abstractmethod
    def get_identifier(self) -> Any:
        raise NotImplementedError

    def __eq__(self, other: Task) -> bool:
        if super().__eq__(other) and hasattr(other, "get_identifier"):
            # additionally compare on the argument level
            return self.get_identifier() == other.get_identifier()
        return False


def get_task(name: str) -> Task:
    try:
        return Task._tasks[name]
    except KeyError as e:
        raise UndefinedTaskError(name) from e


def run_task(name: str, project: ProjectDirectory, **task_kwargs):
    return get_task(name).from_dict(task_kwargs)(project)


@dataclass(frozen=True)
class TaskCrosscorr(Task):

    rr: bool = field(
        default=False,
        metadata=Parameter(
            type=bool,
            help="compute random-random pair counts, even if both randoms are "
                "available"))

    @classmethod
    def get_name(cls) -> str:
        return "cross"

    @classmethod
    def get_help(cls) -> str:
        return "compute the crosscorrelation"

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(cross=self)


@dataclass(frozen=True)
class TaskAutocorr(Task):

    rr: bool = field(
        default=True,
        metadata=Parameter(
            type=bool,
            help="do not compute random-random pair counts"))


@dataclass(frozen=True)
class TaskAutocorrReference(TaskAutocorr):

    @classmethod
    def get_name(cls) -> str:
        return "auto_ref"

    @classmethod
    def get_help(cls) -> str:
        return (
            "compute the reference sample autocorrelation for bias mitigation")

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(auto_ref=self)


@dataclass(frozen=True)
class TaskAutocorrUnknown(TaskAutocorr):

    @classmethod
    def get_name(cls) -> str:
        return "auto_unk"

    @classmethod
    def get_help(cls) -> str:
        return "compute the unknown sample autocorrelation for bias mitigation"

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(auto_unk=self)


@dataclass(frozen=True)
class TaskEstimateCorr(RepeatableTask):

    tag: str = field(
        default="fid",
        metadata=Parameter(
            type=str,
            help="unique identifier for different configurations",
            default_text="(default: %(default)s)"))

    est_cross: str | None = field(
        default=None,
        metadata=Parameter(
            type=str, choices=ESTIMATORS,
            help="correlation estimator for crosscorrelations",
            default_text="(default: LS or DP)",
            parser_id="estimators"))
    est_auto: str | None = field(
        default=None,
        metadata=Parameter(
            type=str, choices=ESTIMATORS,
            help="correlation estimator for autocorrelations",
            default_text="(default: LS or DP)",
            parser_id="estimators"))

    method: str = field(
        default=DEFAULT.Resampling.method,
        metadata=Parameter(
            type=str, choices=METHOD_OPTIONS,
            help="resampling method for covariance estimates",
            default_text="(default: %(default)s)",
            parser_id="sampling"))
    crosspatch: bool = field(
        default=DEFAULT.Resampling.crosspatch,
        metadata=Parameter(
            type=bool,
            help="whether to include cross-patch pair counts when resampling",
            parser_id="sampling"))
    n_boot: int = field(
        default=DEFAULT.Resampling.n_boot,
        metadata=Parameter(
            type=int,
            help="number of bootstrap samples",
            default_text="(default: %(default)s)",
            parser_id="sampling"))
    global_norm: bool = field(
        default=DEFAULT.Resampling.global_norm,
        metadata=Parameter(
            type=bool,
            help="normalise pair counts globally instead of patch-wise",
            parser_id="sampling"))
    seed: int = field(
        default=DEFAULT.Resampling.seed,
        metadata=Parameter(
            type=int,
            help="random seed for bootstrap sample generation",
            default_text="(default: %(default)s)",
            parser_id="sampling"))

    @classmethod
    def get_name(cls) -> str:
        return "zcc"

    @classmethod
    def get_help(cls) -> str:
        return (
            "compute clustering redshift estimates for the unknown data, task "
            "can be added repeatedly if different a 'tag' is used")

    def get_identifier(self) -> str:
        return self.tag

    @property
    def config(self) -> ResamplingConfig:
        return ResamplingConfig(
            method=self.method,
            crosspatch=self.crosspatch,
            n_boot=self.n_boot,
            global_norm=self.global_norm,
            seed=self.seed)

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(zcc=self)


@dataclass(frozen=True)
class TaskTrueRedshifts(Task):

    @classmethod
    def get_name(cls) -> str:
        return "ztrue"

    @classmethod
    def get_help(cls) -> str:
        return (
            "compute true redshift distributions for unknown data (requires "
            "point estimate)")

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(ztrue=self)


@dataclass(frozen=True)
class TaskDropCache(Task):

    @classmethod
    def get_name(cls) -> str:
        return "drop_cache"

    @classmethod
    def get_help(cls) -> str:
        return "delete temporary data in cache directory, has no arguments"

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(drop_cache=self)


@dataclass(frozen=True)
class TaskPlot(Task):

    @classmethod
    def get_name(cls) -> str:
        return "plot"

    @classmethod
    def get_help(cls) -> str:
        return "generate automatic check plots"

    def __call__(self, project: ProjectDirectory) -> Any:
        super().__call__(project)
        project.engine.run(plot=self)


class TaskManager(Sequence):

    def __init__(self) -> None:
        self._tasklist: list[Task] = []

    def __contains__(self, task: Task) -> bool:
        return self._tasklist.__contains__(task)

    def __getitem__(self, item: int) -> Task:
        return self._tasklist.__getitem__(item)

    def __iter__(self) -> Iterator[Task]:
        return self._tasklist.__iter__()

    def __len__(self) -> int:
        return self._tasklist.__len__()

    def __str__(self) -> str:
        tasks = []
        for task in self._tasklist:
            if isinstance(task, RepeatableTask):
                tasks.append(f"{task.get_name()}@{task.get_identifier()}")
            else:
                tasks.append(task.get_name())
        return " > ".join(tasks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def add(self, task: Task) -> None:
        if not isinstance(task, Task):
            raise TypeError(
                f"'task' must be of type {Task}, got {type(task)}")
        try:
            self._tasklist.remove(task)
        except ValueError:
            pass
        bisect.insort(self._tasklist, task)

    def get_tasks(self) -> tuple[Task]:
        return tuple(t for t in self._tasklist)

    @classmethod
    def from_list(
        cls,
        task_list: Sequence[dict[str, Any] | str]
    ) -> TaskManager:
        new = cls()
        for task in task_list:
            if isinstance(task, str):
                name, arg_dict = task, {}
            elif isinstance(task, dict):
                name, arg_dict = task.popitem()
            else:
                raise TaskError(
                    "serialisation format must be 'str(name)' or "
                    "{str(name): dict(**args)}'")
            task_inst = get_task(name).from_dict(arg_dict)
            new.add(task_inst)
        return new

    def to_list(self) -> list[dict[str, Any] | str]:
        task_list = []
        for task in self:
            name = task.get_name()
            arg_dict = task.to_dict()
            if len(arg_dict) == 0:  # no or only default arguments
                task_list.append(name)
            else:
                task_list.append({name: arg_dict})
        return task_list
