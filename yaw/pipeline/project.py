from __future__ import annotations

import logging
import os
import shutil
import textwrap
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from pathlib import Path, _posix_flavour, _windows_flavour
from typing import TYPE_CHECKING, Any, NoReturn

import pandas as pd
import yaml

from yaw import default as DEFAULT
from yaw.catalogs import NewCatalog
from yaw.config import Configuration
from yaw.coordinates import CoordSky
from yaw.utils import DictRepresentation, TypePathStr, bytes_format

from yaw.pipeline.data import InputRegister
from yaw.pipeline.logger import get_logger
from yaw.pipeline.task_utils import TaskList

if TYPE_CHECKING:  # pragma: no cover
    from yaw.catalogs import BaseCatalog
    from yaw.pipeline.data import Input
    from yaw.pipeline.task_utils import TaskRecord


class SetupError(Exception):
    pass


class MissingCatalogError(Exception):
    pass


def _get_numeric_suffix(path: Path) -> int:
    base = path.with_suffix(suffix="").name
    _, num = base.rsplit("_", 1)
    return int(num)


class Directory(Path):

    # seems to be the easiest way to subclass pathlib.Path
    _flavour = _windows_flavour if os.name == 'nt' else _posix_flavour

    def print_contents(self) -> None:
        sizes = dict()
        for path in self.iterdir():
            if path.is_dir():
                sizes[path.name] = ("d", sum(
                    file.stat().st_size for file in path.rglob("*")))
            else:
                sizes[path.name] = ("f", path.stat().st_size)
        print(f"cache path: {self}")
        if len(sizes) == 0:
            width = 10
            total = bytes_format(0)
        else:
            width = min(30, max(len(name) for name in sizes))
            for i, name in enumerate(sorted(sizes), 1):
                print_name = textwrap.shorten(
                    name, width=width, placeholder="...")
                kind, bytes = sizes[name]
                bytes_fmt = bytes_format(bytes)
                print(f"{kind}> {print_name:{width}s}{bytes_fmt:>10s}")
            total = bytes_format(sum(b for _, b in sizes.values()))
        print("-" * (width + 13))
        print(f"{'total':{width+3}s}{total:>10s}")


class CacheDirectory(Directory):

    def _generate_filenames(self, base: str) -> dict[str, Path]:
        tags = ("data", "rand")
        return {tag: Path(self.joinpath(f"{base}.{tag}")) for tag in tags}

    def get_reference(self) -> dict[str, Path]:
        return self._generate_filenames("reference")

    def get_unknown(self, bin_idx: int) -> dict[str, Path]:
        return self._generate_filenames(f"unknown_{bin_idx}")

    def get_bin_indices(self) -> set(int):
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith("unknown"))

    def drop(self, name: str) -> None:
        path = self.joinpath(name)
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()

    def drop_all(self) -> None:
        for path in self.iterdir():
            self.drop(path.name)


class DataDirectory(Directory, ABC):

    @abstractproperty
    def _cross_prefix(self) -> str: pass

    @abstractproperty
    def _auto_prefix(self) -> str: pass

    @abstractmethod
    def get_auto_reference(self) -> Path: pass

    @abstractmethod
    def get_auto(self, bin_idx: int) -> Path: pass

    @abstractmethod
    def get_cross(self, bin_idx: int) -> Path: pass

    def get_cross_indices(self) -> set[int]:
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith(self._cross_prefix))

    def get_auto_indices(self) -> set[int]:
        return set(
            _get_numeric_suffix(path)
            for path in self.iterdir()
            if path.name.startswith(self._auto_prefix))

    def iter_cross(self) -> Iterator[tuple(int, Any)]:
        for idx in sorted(self.get_cross_indices()):
            yield idx, self.get_cross(idx)

    def iter_auto(self) -> Iterator[tuple(int, Any)]:
        for idx in sorted(self.get_auto_indices()):
            yield idx, self.get_auto(idx)


class CountsDirectory(DataDirectory):

    _cross_prefix = "cross"
    _auto_prefix = "auto_unknown"

    def get_auto_reference(self) -> Path:
        return Path(self.joinpath("auto_reference.hdf"))

    def get_auto(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._auto_prefix}_{bin_idx}.hdf"))

    def get_cross(self, bin_idx: int) -> Path:
        return Path(self.joinpath(f"{self._cross_prefix}_{bin_idx}.hdf"))


class EstimateDirectory(DataDirectory):

    _cross_prefix = "nz_unknown"
    _auto_prefix = "auto_unknown"

    def _generate_filenames(self, base: str) -> dict[str, Path]:
        extensions = ("dat", "cov", "boot")
        return {ext: Path(self.joinpath(f"{base}.{ext}")) for ext in extensions}

    def get_auto_reference(self) -> dict[str, Path]:
        return Path(self.joinpath("auto_reference"))

    def get_auto(self, bin_idx: int) -> dict[str, Path]:
        return Path(self.joinpath(f"{self._auto_prefix}_{bin_idx}"))

    def get_cross(self, bin_idx: int) -> dict[str, Path]:
        return Path(self.joinpath(f"{self._cross_prefix}_{bin_idx}"))


def _parse_section_error(exception: Exception, section: str) -> NoReturn:
    if isinstance(exception, KeyError):
        raise SetupError(f"missing section '{section}'") from exception
    raise


def write_setup_file(
    path: TypePathStr,
    setup_dict: dict[str, Any]
) -> None:
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
    setup_dict = load_setup_as_dict(setup_file)
    return parse_config_from_setup(setup_dict)


def parse_config_from_setup(setup_dict: dict[str, Any]) -> Configuration:
    try:
        return Configuration.from_dict(setup_dict["configuration"])
    except KeyError as e:
        _parse_section_error(e, "configuration")


class ProjectDirectory(DictRepresentation):

    def __init__(self, path: TypePathStr) -> None:
        self._path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(
                f"project directory '{self.path}' does not exist")
        if not self.setup_file.exists:
            raise FileNotFoundError(
                f"setup file '{self.setup_file}' does not exist")
        self._add_log_file_handle()
        self.setup_reload()
        # create any missing directories
        self.counts_dir.mkdir(exist_ok=True)
        self.estimate_dir.mkdir(exist_ok=True)

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
        setup_dict = dict(
            configuration=config.to_dict(),
            data=dict(
                cachepath=str(cachepath) if cachepath is not None else None,
                **InputRegister(n_patches).to_dict()),
            backend=backend,
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

    def to_dict(self) -> dict[str, Any]:
        if self._cachepath is None:
            cache_dir = None
        else:
            cache_dir = str(self._cachepath)
        setup = dict(
            configuration=self._config.to_dict(),
            data=dict(
                cachepath=cache_dir,
                **self._inputs.to_dict()),
            backend=self.catalog_factory.backend_name,
            tasks=self._tasks.to_list())
        return setup

    def __enter__(self) -> ProjectDirectory:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # TODO: this is sometimes executed even if an exception was raised
        if exc_type is None:
            self.setup_write()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def log_file(self) -> Path:
        return self._path.joinpath("setup.log")

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("setup.yaml")

    def setup_reload(self) -> None:
        with open(self.setup_file) as f:
            setup = yaml.safe_load(f.read())
        self.catalog_factory = NewCatalog(setup["backend"])
        # configuration is straight forward
        self._config = parse_config_from_setup(setup)
        # set up the data management
        try:
            data = setup["data"]
        except KeyError as e:
            _parse_section_error(e, "data")
        self._cachepath = data.pop("cachepath", None)
        self._cache = CacheDirectory(self.cache_dir)
        self._cache.mkdir(exist_ok=True, parents=True)
        self._inputs = InputRegister.from_dict(data)
        try:
            self._tasks = TaskList.from_list(setup["tasks"])
        except KeyError:
            self._tasks = TaskList()

    def setup_write(self) -> None:
        write_setup_file(self.setup_file, self.to_dict())

    @property
    def config(self) -> Configuration:
        return self._config

    @property
    def patch_file(self) -> Path:
        return self._path.joinpath("patch_centers.csv")

    @property
    def cache_dir(self) -> Path:
        if self._cachepath is None:
            return self._path.joinpath("cache")
        else:
            return Path(self._cachepath)

    def get_cache(self) -> CacheDirectory:
        return self._cache

    def set_reference(
        self,
        data: Input,
        rand: Input | None = None
    ) -> None:
        self._inputs.set_reference(data, rand)
    
    def add_unknown(
        self,
        bin_idx: int,
        data: Input,
        rand: Input | None = None
    ) -> None:
        self._inputs.add_unknown(bin_idx, data, rand)

    def _build_catalog(self, load_kwargs) -> BaseCatalog:
        # patches must be created or applied
        if not self._inputs.external_patches:
            if "patches" in load_kwargs:
                msg = "'n_patches' and catalog 'patches' are mutually exclusive"
                raise SetupError(msg)
            # load and apply existing patch centers
            if self.patch_file.exists():
                centers = pd.read_csv(str(self.patch_file))
                load_kwargs["patches"] = CoordSky(
                    ra=centers["ra"], dec=centers["dec"]).to_3d()
            # schedule patch creation
            else:
                load_kwargs["patches"] = self._inputs.n_patches

        catalog = self.catalog_factory.from_file(**load_kwargs)
        # store patch centers for consecutive loads
        if not self.patch_file.exists():
            centers = pd.DataFrame(dict(
                ra=catalog.centers.ra,
                dec=catalog.centers.dec))
            centers.to_csv(str(self.patch_file), index=False)
        return catalog

    def _load_catalog(
        self,
        sample: str,
        kind: str,
        bin_idx: int | None = None
    ) -> BaseCatalog:
        # get the correct sample type
        if sample == "reference":
            inputs = self._inputs.get_reference()
        elif sample == "unknown":
            if bin_idx is None:
                raise ValueError("no 'bin_idx' provided")
            inputs = self._inputs.get_unknown(bin_idx)
        else:
            raise ValueError("'sample' must be either of 'reference'/'unknown'")
        # get the correct sample kind
        try:
            input = inputs[kind]
        except KeyError as e:
            raise ValueError("'kind' must be either of 'data'/'rand'") from e
        if input is None:
            if kind == "rand":
                kind = "random"
            bin_info = f" bin {bin_idx} " if bin_idx is not None else " "
            raise MissingCatalogError(
                f"no {sample} {kind}{bin_info}catalog specified")

        # get arguments for Catalog.from_file()
        kwargs = input.to_dict()
        kwargs.pop("cache", False)
        # attempt to load the catalog into memory
        if sample == "reference":
            cachepath = self._cache.get_reference()[kind]
        else:
            cachepath = self._cache.get_unknown(bin_idx)[kind]
        exists = cachepath.exists()

        # already cached and backend supports restoring
        if exists and input.cache:
            catalog = self.catalog_factory.from_cache(cachepath)
        # no caching requested or not supported
        else:
            if input.cache:
                kwargs["cache_directory"] = str(cachepath)
                cachepath.mkdir(exist_ok=True)
            catalog = self._build_catalog(kwargs)
        return catalog

    def load_reference(self, kind: str) -> BaseCatalog:
        return self._load_catalog("reference", kind)

    def load_unknown(self, kind: str, bin_idx: int) -> BaseCatalog:
        return self._load_catalog("unknown", kind, bin_idx=bin_idx)

    def get_bin_indices(self) -> set[int]:
        return self._inputs.get_bin_indices()

    @property
    def n_bins(self):
        return self._inputs.n_bins

    def show_catalogs(self) -> None:
        print(yaml.dump(self._inputs.to_dict()))

    @property
    def counts_dir(self) -> Path:
        return self.path.joinpath("paircounts")

    def get_counts(
        self,
        scale_key: str,
        create: bool = False
    ) -> CountsDirectory:
        path = self.counts_dir.joinpath(scale_key)
        if create:
            path.mkdir(exist_ok=True)
        return CountsDirectory(path)

    def list_counts_scales(self) -> list(str):
        return [
            path.name for path in self.counts_dir.iterdir() if path.is_dir()]

    @property
    def estimate_dir(self) -> Path:
        return self.path.joinpath("estimate")

    def get_estimate(
        self,
        scale_key: str,
        create: bool = False
    ) -> EstimateDirectory:
        path = self.estimate_dir.joinpath(scale_key)
        if create:
            path.mkdir(exist_ok=True)
        return EstimateDirectory(path)

    def list_estimate_scales(self) -> list(str):
        return [
            path.name for path in self.estimate_dir.iterdir() if path.is_dir()]

    @property
    def true_dir(self) -> Path:
        return self.path.joinpath("true")

    def get_true_reference(
        self,
        create: bool = False
    ) -> Path:
        if create:
            self.true_dir.mkdir(exist_ok=True)
        return self.true_dir.joinpath(f"nz_reference")

    def get_true_unknown(
        self,
        bin_idx: int,
        create: bool = False
    ) -> Path:
        if create:
            self.true_dir.mkdir(exist_ok=True)
        return self.true_dir.joinpath(f"nztrue_{bin_idx}")

    def get_total_unknown(self) -> Path:
        return self.true_dir.joinpath("bin_counts.dat")

    def add_task(self, task: TaskRecord) -> None:
        self._tasks.add(task)

    def list_tasks(self) -> list[TaskRecord]:
        return list(self._tasks)
