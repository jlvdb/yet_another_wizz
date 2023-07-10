from __future__ import annotations

import bisect
import functools
import logging
from abc import abstractclassmethod, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from yaw.config import OPTIONS, ResamplingConfig
from yaw.config import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.docs import Parameter
from yaw.correlation.estimators import CorrelationEstimator
from yaw_cli.pipeline.logger import print_yaw_message
from yaw_cli.pipeline.processing import DataProcessor

if TYPE_CHECKING:  # pragma: no cover
    from argparse import Namespace

    from yaw_cli.pipeline.project import ProjectDirectory


logger = logging.getLogger(__name__)

ESTIMATORS = [est.short for est in CorrelationEstimator.variants]


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


class MergedTask(Task):
    pass


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


@dataclass(frozen=True)
class TaskCrosscorr(Task):
    rr: bool = field(
        default=False,
        metadata=Parameter(
            type=bool,
            help="compute random-random pair counts, even if both randoms are available",
        ),
    )

    @classmethod
    def get_name(cls) -> str:
        return "cross"

    @classmethod
    def get_help(cls) -> str:
        return "compute the crosscorrelation"


@dataclass(frozen=True)
class TaskAutocorr(Task):
    rr: bool = field(
        default=True,
        metadata=Parameter(type=bool, help="do not compute random-random pair counts"),
    )


@dataclass(frozen=True)
class TaskAutocorrReference(TaskAutocorr):
    @classmethod
    def get_name(cls) -> str:
        return "auto_ref"

    @classmethod
    def get_help(cls) -> str:
        return "compute the reference sample autocorrelation for bias mitigation"


@dataclass(frozen=True)
class TaskAutocorrUnknown(TaskAutocorr):
    @classmethod
    def get_name(cls) -> str:
        return "auto_unk"

    @classmethod
    def get_help(cls) -> str:
        return "compute the unknown sample autocorrelation for bias mitigation"


@dataclass(frozen=True)
class TaskTrueRedshifts(Task):
    @classmethod
    def get_name(cls) -> str:
        return "ztrue"

    @classmethod
    def get_help(cls) -> str:
        return (
            "compute true redshift distributions for unknown data (requires "
            "point estimate)"
        )

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


@dataclass(frozen=True)
class TaskEstimateCorr(MergedTask, RepeatableTask):
    tag: str = field(
        default="fid",
        metadata=Parameter(
            type=str,
            help="unique identifier for different configurations",
            default_text="(default: %(default)s)",
        ),
    )
    bias_ref: bool = field(
        default=True,
        metadata=Parameter(
            type=bool,
            help="whether to mitigate the reference sample bias using its "
            "autocorrelation function (if available)",
        ),
    )
    bias_unk: bool = field(
        default=True,
        metadata=Parameter(
            type=bool,
            help="whether to mitigate the unknown sample bias using its "
            "autocorrelation functions (if available)",
        ),
    )

    est_cross: str | None = field(
        default=None,
        metadata=Parameter(
            type=str,
            choices=ESTIMATORS,
            help="correlation estimator for crosscorrelations",
            default_text="(default: LS or DP)",
            parser_id="estimators",
        ),
    )
    est_auto: str | None = field(
        default=None,
        metadata=Parameter(
            type=str,
            choices=ESTIMATORS,
            help="correlation estimator for autocorrelations",
            default_text="(default: LS or DP)",
            parser_id="estimators",
        ),
    )

    method: str = field(
        default=DEFAULT.Resampling.method,
        metadata=Parameter(
            type=str,
            choices=OPTIONS.method,
            help="resampling method for covariance estimates",
            default_text="(default: %(default)s)",
            parser_id="sampling",
        ),
    )
    crosspatch: bool = field(
        default=DEFAULT.Resampling.crosspatch,
        metadata=Parameter(
            type=bool,
            help="whether to include cross-patch pair counts when resampling",
            parser_id="sampling",
        ),
    )
    n_boot: int = field(
        default=DEFAULT.Resampling.n_boot,
        metadata=Parameter(
            type=int,
            help="number of bootstrap samples",
            default_text="(default: %(default)s)",
            parser_id="sampling",
        ),
    )
    global_norm: bool = field(
        default=DEFAULT.Resampling.global_norm,
        metadata=Parameter(
            type=bool,
            help="normalise pair counts globally instead of patch-wise",
            parser_id="sampling",
        ),
    )
    seed: int = field(
        default=DEFAULT.Resampling.seed,
        metadata=Parameter(
            type=int,
            help="random seed for bootstrap sample generation",
            default_text="(default: %(default)s)",
            parser_id="sampling",
        ),
    )

    @classmethod
    def get_name(cls) -> str:
        return "zcc"

    @classmethod
    def get_help(cls) -> str:
        return (
            "compute clustering redshift estimates for the unknown data, task "
            "can be added repeatedly if different a 'tag' is used"
        )

    def get_identifier(self) -> str:
        return self.tag

    @property
    def config(self) -> ResamplingConfig:
        return ResamplingConfig(
            method=self.method,
            crosspatch=self.crosspatch,
            n_boot=self.n_boot,
            global_norm=self.global_norm,
            seed=self.seed,
        )


@dataclass(frozen=True)
class TaskPlot(MergedTask, Task):
    @classmethod
    def get_name(cls) -> str:
        return "plot"

    @classmethod
    def get_help(cls) -> str:
        return "generate automatic check plots"


class TaskManager(Sequence):
    def __init__(self, project: ProjectDirectory) -> None:
        self._engine = DataProcessor(project)
        self._history: list[Task] = []
        self._queue: deque[Task] = deque([])

    @classmethod
    def from_history_list(
        cls, task_list: Sequence[dict[str, Any] | str], project: ProjectDirectory
    ) -> TaskManager:
        new = cls(project)
        for task in task_list:
            if isinstance(task, str):
                name, arg_dict = task, {}
            elif isinstance(task, dict):
                name, arg_dict = task.popitem()
            else:
                raise TaskError(
                    "serialisation format must be 'str(name)' or {str(name): dict(**args)}'"
                )
            task_inst = get_task(name).from_dict(arg_dict)
            new._insert_task(task_inst, new._history)
        return new

    def history_to_list(self) -> list[dict[str, Any] | str]:
        task_list = []
        for task in self._history:
            name = task.get_name()
            arg_dict = task.to_dict()
            if len(arg_dict) == 0:  # no or only default arguments
                task_list.append(name)
            else:
                task_list.append({name: arg_dict})
        return task_list

    def __contains__(self, task: Task) -> bool:
        return self._queue.__contains__(task)

    def __getitem__(self, item: int) -> Task:
        return self._queue.__getitem__(item)

    def __iter__(self) -> Iterator[Task]:
        return self._queue.__iter__()

    def __len__(self) -> int:
        return self._queue.__len__()

    def _view(self, task_list: list[Task]) -> str:
        tasks = []
        for task in task_list:
            if isinstance(task, RepeatableTask):
                tasks.append(f"{task.get_name()}@{task.get_identifier()}")
            else:
                tasks.append(task.get_name())
        return " > ".join(tasks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.view_queue()})"

    def view_queue(self) -> str:
        return self._view(self._queue)

    def view_history(self) -> str:
        return self._view(self._history)

    def get_queued(self) -> tuple[Task]:
        return tuple(t for t in self._queue)

    def get_history(self) -> tuple[Task]:
        return tuple(t for t in self._history)

    def _insert_task(self, task: Task, task_list: list[Task]) -> None:
        if not isinstance(task, Task):
            raise TypeError(f"'task' must be of type {Task}, got {type(task)}")
        try:
            task_list.remove(task)
        except ValueError:
            pass
        bisect.insort(task_list, task)

    def schedule(self, task: Task) -> None:
        t_args = ", ".join(f"{k}={repr(v)}" for k, v in asdict(task).items())
        if len(t_args) == 0:
            t_args = "---"
        logger.debug(f"'{task.get_name()}' arguments: {t_args}")
        self._insert_task(task, self._queue)

    def reschedule_history(self) -> None:
        for task in self.get_history():
            self.schedule(task)

    def drop(self, task: Task) -> Task:
        self._queue.remove(task)
        return task

    def clear(self) -> None:
        self._queue.clear()

    def process(self, progress: bool = False, threads: int | None = None) -> None:
        kwargs = self._get_run_args(self._queue)
        try:
            self._engine.set_run_context(progress=progress, threads=threads)
            self._run(**kwargs)
        finally:
            self._engine.reset_run_context()
        # move tasks to history
        while len(self._queue) > 0:
            task = self._queue.pop()
            self._insert_task(task, self._history)

    def run(
        self, task: Task, progress: bool = False, threads: int | None = None
    ) -> None:
        # log task
        logger.info(f"running task '{task.get_name()}'")
        args = ", ".join(f"{k}={repr(v)}" for k, v in asdict(task).items())
        if len(args) == 0:
            args = "---"
        logger.debug(f"arguments: {args}")
        # use a temporary single item queue
        queue = self._queue
        try:
            self._queue = deque([])
            self._insert_task(task, self._queue)
            self.process(progress=progress, threads=threads)
        finally:
            self._queue = queue

    def _get_run_args(
        self, task_list: list[Task]
    ) -> dict[str, Task | list[RepeatableTask]]:
        tasks = {}
        for task in task_list:
            name = task.get_name()
            # handle repeatable task
            if name in tasks:
                if isinstance(tasks[name], RepeatableTask):
                    tasks[name] = [tasks[name]]
                tasks[name].append(task)
            else:
                tasks[name] = task
        return tasks

    def _run(
        self,
        cross: TaskCrosscorr | None = None,
        auto_ref: TaskAutocorrReference | None = None,
        auto_unk: TaskAutocorrUnknown | None = None,
        zcc: Sequence[TaskEstimateCorr] | TaskEstimateCorr | None = None,
        ztrue: TaskTrueRedshifts | None = None,
        drop_cache: TaskDropCache | None = None,
        plot: TaskPlot | None = None,
    ) -> None:
        engine = self._engine

        do_w_sp = cross is not None
        do_w_ss = auto_ref is not None
        do_w_pp = auto_unk is not None
        do_zcc = zcc is not None
        do_true = ztrue is not None
        if not do_zcc:
            zcc = tuple()
        elif isinstance(zcc, Sequence):
            zcc = tuple(zcc)
        else:
            zcc = (zcc,)

        # some state parameters
        zcc_processed = False
        state = engine.state
        has_w_ss = state.has_w_ss
        has_w_sp = state.has_w_sp
        has_w_pp = state.has_w_pp

        if do_w_sp or do_w_ss or (do_zcc and has_w_ss):
            print_yaw_message("processing reference sample")
        if do_w_sp or do_w_ss:
            engine.load_reference()
            engine.write_nz_ref()

        if do_w_ss:
            engine.compute_linkage()
            engine.run_auto_ref(compute_rr=auto_ref.rr)
        elif do_zcc and has_w_ss:
            engine.load_auto_ref()
        if do_zcc and engine._w_ss is not None:
            for zcc_task in zcc:
                engine.sample_auto_ref(
                    tag=zcc_task.tag,
                    config=zcc_task.config,
                    estimator=zcc_task.est_auto,
                )
                engine.write_auto_ref(zcc_task.tag)
                zcc_processed = True

        if do_w_sp or do_w_pp or (do_zcc and (has_w_sp or has_w_pp)) or do_true:
            for i, idx in enumerate(engine.iter_bins(), 1):
                message = "processing unknown "
                if engine.project.n_bins == 1:
                    message += "sample"
                else:
                    message += f"bin {i} / {engine.project.n_bins}"
                print_yaw_message(message)

                if do_w_sp or do_w_pp or do_true:
                    skip_rand = do_true and not (do_w_sp or do_w_pp)
                    engine.load_unknown(skip_rand=skip_rand)

                if do_w_sp:
                    engine.compute_linkage()
                    engine.run_cross(compute_rr=cross.rr)
                    engine.write_total_unk()
                elif do_zcc and has_w_sp:
                    engine.load_cross()

                if do_w_pp:
                    engine.compute_linkage()
                    engine.run_auto_unk(compute_rr=auto_unk.rr)
                    engine.write_total_unk()
                elif do_zcc and has_w_pp:
                    engine.load_auto_unk()

                if do_zcc:
                    for zcc_task in zcc:
                        if engine._w_pp is not None:
                            engine.sample_auto_unk(
                                tag=zcc_task.tag,
                                config=zcc_task.config,
                                estimator=zcc_task.est_auto,
                            )
                            engine.write_auto_unk(tag=zcc_task.tag)
                            zcc_processed = True
                        if engine._w_sp is not None:
                            engine.sample_cross(
                                tag=zcc_task.tag,
                                config=zcc_task.config,
                                estimator=zcc_task.est_cross,
                            )
                            engine.write_nz_cc(
                                tag=zcc_task.tag,
                                bias_ref=zcc_task.bias_ref,
                                bias_unk=zcc_task.bias_unk,
                            )
                            zcc_processed = True

                if do_true:
                    engine.write_nz_true()

        if do_zcc and not zcc_processed:
            logger.warn("task 'zcc': there were no pair counts to process")

        if drop_cache:
            engine.drop_cache()

        if plot:
            print_yaw_message("plotting data")
            engine.plot()
