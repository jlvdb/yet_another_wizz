from __future__ import annotations

import logging
import os
import shutil
from abc import abstractmethod, abstractproperty
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from yaw import __version__
from yaw.config import Configuration
from yaw.config import default as DEFAULT
from yaw.config.utils import parse_section_error
from yaw.core.abc import DictRepresentation
from yaw.core.utils import TypePathStr
from yaw_cli.pipeline.data import InputManager
from yaw_cli.pipeline.directories import (
    CacheDirectory,
    CountsDirectory,
    EstimateDirectory,
    TrueDirectory,
)
from yaw_cli.pipeline.logger import get_logger
from yaw_cli.pipeline.processing import DataProcessor, PostProcessor
from yaw_cli.pipeline.tasks import TaskManager

if TYPE_CHECKING:  # pragma: no cover
    from yaw_cli.pipeline.data import Input


logger = logging.getLogger(__name__)


class SetupError(Exception):
    pass


def check_version(version: str) -> None:
    msg = "configuration was generated on code version different from installed:"
    msg += f" {version} != {__version__}"
    # compare first two digits, which may introduce breaking changes
    this = [int(s) for s in __version__.split(".")][:2]
    other = [int(s) for s in version.split(".")][:2]
    for t, o in zip_longest(this, other, fillvalue=0):
        if t != o:
            raise SetupError(msg)


def compress_config(config: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    compressed = dict(**config)
    for key, value in config.items():
        if key in default:
            refval = default[key]
            if isinstance(value, dict) and hasattr(refval, "__dict__"):
                new = compress_config(value, refval.__dict__)
                if len(new) == 0:
                    compressed.pop(key)
                else:
                    compressed[key] = new
            elif value == refval:
                compressed.pop(key)
    return compressed


def write_setup_file(path: TypePathStr, setup_dict: dict[str, Any]) -> None:
    logger.debug("writing setup file")
    setup_dict = {k: v for k, v in setup_dict.items()}
    setup_dict["_version"] = __version__
    lines = yaml.dump(setup_dict).split("\n")
    # some postprocessing for better readibility
    indent = " " * 4
    string = "# yet_another_wizz setup configuration (auto generated)\n"
    for i, line in enumerate(lines):
        stripped = line.strip(" ")
        if len(stripped) == 0:
            continue
        n_spaces = len(line) - len(stripped)
        n_indent = n_spaces // 2
        is_list = stripped.startswith("- ")
        if n_indent == 0 and not is_list:
            string += f"\n{line}\n"
        else:
            list_indent = "  " if is_list else ""
            string += f"{indent * n_indent}{list_indent}{stripped}\n"
    # write to setup file
    with open(path, "w") as f:
        f.write(string)


def substitute_env_var(setup_dict: dict) -> dict:
    for key, value in setup_dict.items():
        if isinstance(value, str) and value.startswith("$"):
            setup_dict[key] = os.environ[value[1:]]
        elif isinstance(value, dict):
            setup_dict[key] = substitute_env_var(value)
    return setup_dict


def load_setup_as_dict(setup_file: TypePathStr) -> dict[str, Any]:
    with open(setup_file) as f:
        try:
            return yaml.safe_load(f.read())
        except Exception as e:
            raise SetupError(f"parsing the setup file '{setup_file}' failed") from e


def load_config_from_setup(setup_file: TypePathStr) -> Configuration:
    logger.info(f"importing configuration from '{setup_file}'")
    setup_dict = load_setup_as_dict(setup_file)
    return parse_config_from_setup(setup_dict)


def parse_config_from_setup(setup_dict: dict[str, Any]) -> Configuration:
    try:
        return Configuration.from_dict(setup_dict["configuration"])
    except KeyError as e:
        parse_section_error(e, "configuration", SetupError)


@dataclass(frozen=True)
class ProjectState:
    has_reference: bool = False
    has_unknown: bool = False
    has_w_ss: bool = False
    has_w_sp: bool = False
    has_w_pp: bool = False
    has_w_ss_cf: bool = False
    has_w_pp_cf: bool = False
    has_nz_cc: bool = False
    has_nz_ref: bool = False
    has_nz_true: bool = False


class YawDirectory(DictRepresentation):
    _tasks: TaskManager

    def __init__(self, path: TypePathStr) -> None:
        self._path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"project directory '{self.path}' does not exist")
        if not self.setup_file.exists():
            raise FileNotFoundError(
                f"not a {self.__class__.__name__}, setup file "
                f"'{self.setup_file}' does not exist"
            )
        if not self.log_file.exists():
            logger.info(f"setting up log file '{self.log_file}'")
        else:
            logger.info(f"resuming project at '{self._path}'")
        self._add_log_file_handle()
        with open(self.setup_file) as f:
            setup = yaml.safe_load(f.read())
        setup = substitute_env_var(setup)
        self.setup_reload(setup)
        # create any missing directories
        self.counts_path.mkdir(exist_ok=True)
        self.estimate_path.mkdir(exist_ok=True)
        self.get_true_dir().mkdir(exist_ok=True)

    def _add_log_file_handle(self):
        # create the file logger
        logger = get_logger()
        fh = logging.FileHandler(self.log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def __enter__(self) -> ProjectDirectory:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # TODO: this is sometimes executed even if an exception was raised
        if exc_type is None:
            self.setup_write()

    @abstractproperty
    def setup_file(self) -> Path:
        pass

    def setup_write(self) -> None:
        write_setup_file(self.setup_file, self.to_dict())

    @abstractmethod
    def setup_reload(self, setup: dict) -> None:
        check_version(setup.pop("_version", __version__))
        # configuration is straight forward
        self._config = parse_config_from_setup(setup)

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        path: TypePathStr,
    ) -> YawDirectory:
        new = cls.__new__(cls)  # access to path attributes
        new._path = Path(path).expanduser()
        new._path.mkdir(parents=True, exist_ok=False)
        # create the setup file
        write_setup_file(new.setup_file, the_dict)
        return cls(path)

    @property
    def config(self) -> Configuration:
        return self._config

    def iter_scales(self) -> Iterator[str]:
        for scale in self.config.scales:
            yield str(scale)

    def get_state(self) -> ProjectState:
        # input data
        try:
            has_reference = self.inputs.has_reference
            has_unknown = self.inputs.has_unknown
        except AttributeError:
            has_reference = False
            has_unknown = False
        # pair counts
        try:
            _, counts_dir = next(self.iter_counts())
            has_w_ss = counts_dir.has_auto_reference
            has_w_sp = counts_dir.has_cross
            has_w_pp = counts_dir.has_auto
        except StopIteration:
            has_w_ss = False
            has_w_sp = False
            has_w_pp = False
        # samples correlation functions
        try:
            _, est_dir = next(self.iter_estimate())
            has_w_ss_cf = est_dir.has_auto_reference
            has_w_pp_cf = est_dir.has_auto
            has_nz_cc = est_dir.has_cross
        except StopIteration:
            has_w_ss_cf = False
            has_w_pp_cf = False
            has_nz_cc = False
        true_dir = self.get_true_dir()
        return ProjectState(
            has_reference=has_reference,
            has_unknown=has_unknown,
            has_w_ss=has_w_ss,
            has_w_sp=has_w_sp,
            has_w_pp=has_w_pp,
            has_w_ss_cf=has_w_ss_cf,
            has_w_pp_cf=has_w_pp_cf,
            has_nz_cc=has_nz_cc,
            has_nz_ref=true_dir.has_reference,
            has_nz_true=true_dir.has_unknown,
        )

    @property
    def processor(self) -> PostProcessor:
        return self._tasks._engine

    @property
    def path(self) -> Path:
        return self._path

    @property
    def log_file(self) -> Path:
        return self._path.joinpath("setup.log")

    @property
    def patch_file(self) -> Path:
        return self._path.joinpath("patch_centers.dat")

    @property
    def bin_weight_file(self) -> Path:
        return self._path.joinpath("bin_weights.dat")

    @property
    def counts_path(self) -> Path:
        return self.path.joinpath("paircounts")

    def get_counts_dir(self, scale_key: str, create: bool = False) -> CountsDirectory:
        path = self.counts_path.joinpath(scale_key)
        if create:
            path.mkdir(exist_ok=True)
        return CountsDirectory(path)

    def iter_counts(
        self, create: bool = False
    ) -> Iterator[tuple[str, CountsDirectory]]:
        for scale in self.iter_scales():
            counts = self.get_counts_dir(scale, create=create)
            yield scale, counts

    @property
    def estimate_path(self) -> Path:
        return self.path.joinpath("estimate")

    def get_estimate_dir(
        self, scale_key: str, tag: str = "fid", create: bool = False
    ) -> EstimateDirectory:
        path = self.estimate_path.joinpath(scale_key, tag)
        if create:
            path.mkdir(exist_ok=True, parents=True)
        return EstimateDirectory(path)

    def iter_tags(self) -> Iterator[str]:
        tags = set()
        for scale in self.iter_scales():
            path = self.estimate_path.joinpath(scale)
            if path.exists():
                for tag_path in path.iterdir():
                    if tag_path.is_dir():
                        tags.add(tag_path.name)
        for tag in sorted(tags):
            yield tag

    def iter_estimate(
        self, create: bool = False, tag: str = DEFAULT.NotSet
    ) -> Iterator[tuple[tuple[str, str], EstimateDirectory]]:
        if tag is DEFAULT.NotSet:
            tag_iter = self.iter_tags()
        else:
            tag_iter = (tag,)
        for tag in tag_iter:
            for scale in self.iter_scales():
                est = self.get_estimate_dir(scale, tag, create=create)
                yield (scale, tag), est

    def get_true_dir(self, create: bool = False) -> TrueDirectory:
        path = self.path.joinpath("true")
        if create:
            path.mkdir(exist_ok=True)
        return TrueDirectory(path)

    @property
    def tasks(self) -> TaskManager:
        return self._tasks

    @abstractmethod
    def get_bin_indices(self) -> set[int]:
        pass

    @abstractproperty
    def n_bins(self) -> int:
        pass


class ProjectDirectory(YawDirectory):
    @classmethod
    def create(
        cls,
        path: TypePathStr,
        config: Configuration,
        n_patches: int | None = None,
        cachepath: TypePathStr | None = None,
        backend: str = DEFAULT.backend,
    ) -> ProjectDirectory:
        logger.info(f"creating new project at '{path}'")
        data = dict()
        if n_patches is not None:
            data["n_patches"] = n_patches
        if cachepath is not None:
            data["cachepath"] = cachepath
        if backend != DEFAULT.backend:
            data["backend"] = backend
        setup_dict = dict(configuration=config.to_dict(), data=data, tasks=[])
        return cls.from_dict(setup_dict, path=path)

    @classmethod
    def from_setup(cls, path: TypePathStr, setup_file: TypePathStr) -> ProjectDirectory:
        new = cls.__new__(cls)  # access to path attributes
        new._path = Path(path).expanduser()
        new._path.mkdir(parents=True, exist_ok=False)
        # copy setup file
        shutil.copy(str(setup_file), str(new.setup_file))
        return cls(path)

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("setup.yaml")

    def setup_reload(self, setup: dict) -> None:
        super().setup_reload(setup)
        # set up task management
        task_list = setup.get("tasks", [])
        self._tasks = TaskManager.from_history_list(task_list, project=self)
        # set up the data management
        try:
            data = setup["data"]
        except KeyError as e:
            parse_section_error(e, "data", SetupError)
        # cache needs extra care: if None, set to default location
        if "cachepath" not in data or data["cachepath"] is None:
            data["cachepath"] = str(self.default_cache_path)
        else:
            logger.info(f"using cache location '{data['cachepath']}'")
        self._inputs = InputManager.from_dict(data)
        # try loading existsing patch centers
        if self.patch_file.exists():
            self._inputs.centers_from_file(self.patch_file)

    def to_dict(self) -> dict[str, Any]:
        # strip default values from config
        configuration = compress_config(
            self._config.to_dict(), DEFAULT.Configuration.__dict__
        )
        setup = dict(
            configuration=configuration,
            data=self._inputs.to_dict(),
            tasks=self._tasks.history_to_list(),
        )
        # cache: if default location set to None. Reason: if cloning setup,
        # original cache would be used (unless manually overridden)
        if setup["data"]["cachepath"] == str(self.default_cache_path):
            setup["data"].pop("cachepath")
        return setup

    @property
    def inputs(self) -> InputManager:
        return self._inputs

    @property
    def processor(self) -> DataProcessor:
        return self._tasks._engine

    @property
    def default_cache_path(self) -> Path:
        return self._path.joinpath("cache")

    def get_cache_dir(self) -> CacheDirectory:
        return self.inputs.get_cache()

    def set_reference(self, data: Input, rand: Input | None = None) -> None:
        if rand is not None:
            logger.debug(f"registering reference random catalog '{rand.filepath}'")
        self._inputs.set_reference(data, rand)

    def add_unknown(self, bin_idx: int, data: Input, rand: Input | None = None) -> None:
        if rand is not None:
            logger.debug(
                f"registering unknown bin {bin_idx} random catalog "
                f"'{rand.filepath}'"
            )
        self._inputs.add_unknown(bin_idx, data, rand)

    def get_bin_indices(self) -> set[int]:
        return self._inputs.get_bin_indices()

    @property
    def n_bins(self) -> int:
        return self._inputs.n_bins
