from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from yaw import default as DEFAULT
from yaw.config import Configuration, parse_section_error
from yaw.utils import DictRepresentation, TypePathStr
from yaw.pipeline.data import InputManager
from yaw.pipeline.directories import (
    CacheDirectory, CountsDirectory, EstimateDirectory, TrueDirectory)
from yaw.pipeline.logger import get_logger
from yaw.pipeline.task_utils import TaskList

if TYPE_CHECKING:  # pragma: no cover
    from yaw.catalogs import BaseCatalog
    from yaw.pipeline.data import Input
    from yaw.pipeline.task_utils import TaskRecord


logger = logging.getLogger(__name__)


class SetupError(Exception):
    pass


def write_setup_file(
    path: TypePathStr,
    setup_dict: dict[str, Any]
) -> None:
    logger.debug("writing setup file")
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


def load_setup_as_dict(setup_file: TypePathStr) -> dict[str, Any]:
    with open(setup_file) as f:
        try:
            return yaml.safe_load(f.read())
        except Exception as e:
            raise SetupError(
                f"parsing the setup file '{setup_file}' failed") from e


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
    has_reference: bool
    has_unknown: bool
    has_w_ss: bool
    has_w_sp: bool
    has_w_pp: bool
    has_w_ss_cf: bool
    has_w_pp_cf: bool
    has_nz_cc: bool
    has_nz_ref: bool
    has_nz_true: bool


class ProjectDirectory(DictRepresentation):

    def __init__(self, path: TypePathStr) -> None:
        self._path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(
                f"project directory '{self.path}' does not exist")
        if not self.setup_file.exists:
            raise FileNotFoundError(
                f"setup file '{self.setup_file}' does not exist")
        if not self.log_file.exists():
            logger.info(f"setting up log file '{self.log_file}'")
        else:
            logger.info(f"resuming project at '{self._path}'")
        self._add_log_file_handle()
        self.setup_reload()
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
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        path: TypePathStr,
    ) -> ProjectDirectory:
        new = cls.__new__(cls)  # access to path attributes
        new._path = Path(path).expanduser()
        new._path.mkdir(parents=True, exist_ok=False)
        # create the setup file
        write_setup_file(new.setup_file, the_dict)
        return cls(path)

    @classmethod
    def create(
        cls,
        path: TypePathStr,
        config: Configuration,
        n_patches: int | None = None,
        cachepath: TypePathStr | None = None,
        backend: str = DEFAULT.backend
    ) -> ProjectDirectory:
        logger.info(f"creating new project at '{path}'")
        data = dict()
        if n_patches is not None:
            data["n_patches"] = n_patches
        if cachepath is not None:
            data["cachepath"] = cachepath
        if backend != DEFAULT.backend:
            data["backend"] = backend
        setup_dict = dict(
            configuration=config.to_dict(),
            data=data,
            tasks=TaskList().to_list())
        return cls.from_dict(setup_dict, path=path)

    @classmethod
    def from_setup(
        cls,
        path: TypePathStr,
        setup_file: TypePathStr
    ) -> ProjectDirectory:
        new = cls.__new__(cls)  # access to path attributes
        new._path = Path(path).expanduser()
        new._path.mkdir(parents=True, exist_ok=False)
        # copy setup file
        shutil.copy(str(setup_file), str(new.setup_file))
        return cls(path)

    def setup_reload(self) -> None:
        with open(self.setup_file) as f:
            setup = yaml.safe_load(f.read())
        # configuration is straight forward
        self._config = parse_config_from_setup(setup)
        # set up the data management
        try:
            data = setup["data"]
        except KeyError as e:
            parse_section_error(e, "data", SetupError)
        # cache needs extra care: if None, set to default location
        if "cachepath" not in data or data["cachepath"] is None:
            data["cachepath"] = str(self.default_cache_path)
        self._inputs = InputManager.from_dict(data)
        # try loading existsing patch centers
        if self.patch_file.exists():
            self._inputs.centers_from_file(self.patch_file)
        # set up task management
        try:
            self._tasks = TaskList.from_list(setup["tasks"])
        except KeyError:
            self._tasks = TaskList()

    def to_dict(self) -> dict[str, Any]:
        setup = dict(
            configuration=self._config.to_dict(),
            data=self._inputs.to_dict(),
            tasks=self._tasks.to_list())
        # cache: if default location set to None. Reason: if cloning setup,
        # original cache would be used (unless manually overridden)
        if setup["data"]["cachepath"] == str(self.default_cache_path):
            setup["data"].pop("cachepath")
        return setup

    def setup_write(self) -> None:
        write_setup_file(self.setup_file, self.to_dict())

    def __enter__(self) -> ProjectDirectory:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # TODO: this is sometimes executed even if an exception was raised
        if exc_type is None:
            self.setup_write()

    def get_state(self) -> ProjectState:
        _, counts_dir = next(self.iter_counts())
        _, est_dir = next(self.iter_estimate())
        true_dir = self.get_true_dir()
        return ProjectState(
            has_reference=self.inputs.has_reference,
            has_unknown=self.inputs.has_unknown,
            has_w_ss=counts_dir.has_auto_reference,
            has_w_sp=counts_dir.has_cross,
            has_w_pp=counts_dir.has_auto,
            has_w_ss_cf=est_dir.has_auto_reference,
            has_w_pp_cf=est_dir.has_auto,
            has_nz_cc=est_dir.has_cross,
            has_nz_ref=true_dir.has_reference,
            has_nz_true=true_dir.has_unknown)

    def iter_scales(self) -> Iterator[str]:
        for scale in self.config.scales.dict_keys():
            yield scale

    @property
    def path(self) -> Path:
        return self._path

    @property
    def log_file(self) -> Path:
        return self._path.joinpath("setup.log")

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("setup.yaml")

    @property
    def config(self) -> Configuration:
        return self._config

    @property
    def inputs(self) -> InputManager:
        return self._inputs

    @property
    def default_cache_path(self) -> Path:
        return self._path.joinpath("cache")

    def get_cache_dir(self) -> CacheDirectory:
        return self.inputs.get_cache()

    @property
    def patch_file(self) -> Path:
        return self._path.joinpath("patch_centers.dat")

    @property
    def bin_weight_file(self) -> Path:
        return self._path.joinpath("bin_weights.dat")

    def set_reference(
        self,
        data: Input,
        rand: Input | None = None
    ) -> None:
        if rand is not None:
            logger.debug(
                f"registering reference random catalog '{rand.filepath}'")
        self._inputs.set_reference(data, rand)
    
    def add_unknown(
        self,
        bin_idx: int,
        data: Input,
        rand: Input | None = None
    ) -> None:
        if rand is not None:
            logger.debug(
                f"registering unknown bin {bin_idx} random catalog "
                f"'{rand.filepath}'")
        self._inputs.add_unknown(bin_idx, data, rand)

    def load_reference(
            self,
        kind: str,
        progress: bool = False
    ) -> BaseCatalog:
        cat = self._inputs.load_reference(kind=kind, progress=progress)
        if not self.patch_file.exists():
            self._inputs.centers_to_file(self.patch_file)
        return cat

    def load_unknown(
        self,
        kind: str,
        bin_idx: int,
        progress: bool = False
    ) -> BaseCatalog:
        cat = self._inputs.load_unknown(
            kind=kind, bin_idx=bin_idx, progress=progress)
        if not self.patch_file.exists():
            self._inputs.centers_to_file(self.patch_file)
        return cat

    def get_bin_indices(self) -> set[int]:
        return self._inputs.get_bin_indices()

    @property
    def n_bins(self):
        return self._inputs.n_bins

    @property
    def counts_path(self) -> Path:
        return self.path.joinpath("paircounts")

    def get_counts_dir(
        self,
        scale_key: str,
        create: bool = False
    ) -> CountsDirectory:
        path = self.counts_path.joinpath(scale_key)
        if create:
            path.mkdir(exist_ok=True)
        return CountsDirectory(path)

    def iter_counts(
        self,
        create: bool = False
    ) -> Iterator[tuple[str, CountsDirectory]]:
        for scale in self.iter_scales():
            counts = self.get_counts_dir(scale, create=create)
            yield scale, counts

    @property
    def estimate_path(self) -> Path:
        return self.path.joinpath("estimate")

    def get_estimate_dir(
        self,
        scale_key: str,
        create: bool = False
    ) -> EstimateDirectory:
        path = self.estimate_path.joinpath(scale_key)
        if create:
            path.mkdir(exist_ok=True)
        return EstimateDirectory(path)

    def iter_estimate(
        self,
        create: bool = False
    ) -> Iterator[tuple[str, EstimateDirectory]]:
        for scale in self.iter_scales():
            counts = self.get_estimate_dir(scale, create=create)
            yield scale, counts

    def get_true_dir(
        self,
        create: bool = False
    ) -> TrueDirectory:
        path = self.path.joinpath("true")
        if create:
            path.mkdir(exist_ok=True)
        return TrueDirectory(path)

    def add_task(self, task: TaskRecord) -> None:
        self._tasks.add(task)

    def list_tasks(self) -> list[TaskRecord]:
        return list(self._tasks)
