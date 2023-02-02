from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NoReturn

import yaml

from yaw.core.config import Configuration

from yaw.pipe.data import InputRegister
from yaw.pipe.directories import (
    CacheDirectory, CountsDirectory, EstimateDirectory)

if TYPE_CHECKING:
    from yaw.core.catalog import CatalogBase
    from yaw.pipe.data import Input


class InvalidBackendError(Exception):
    pass


class SetupError(Exception):
    pass


class MissingCatalogError(Exception):
    pass


def _parse_section_error(exception: Exception, section: str) -> NoReturn:
    if isinstance(exception, KeyError):
        raise SetupError(f"missing section '{section}'") from exception
    raise


class Setup:

    def __init__(self, setupfile: Path | str) -> None:
        # read the setup file
        self._path = str(setupfile)
        self.reload()

    def reload(self) -> None:
        with open(self.path) as f:
            setup = yaml.safe_load(f.read())
        # get the backend first which is used to import the backend package
        self.backend_name = setup["backend"]
        self._backend = None
        self.backend  # try to import
        # configuration is straight forward
        try:
            self.config = Configuration.from_dict(setup["configuration"])
        except KeyError as e:
            _parse_section_error(e, "configuration")
        # set up the data management
        try:
            data = setup["data"]
        except KeyError as e:
            _parse_section_error(e, "data")
        cachepath = data.get("cachepath")
        if cachepath is None:
            self.cache = None
        else:
            self.cache = CacheDirectory(cachepath)
            self.cache.mkdir(exist_ok=True, parents=True)
        self.catalogs = InputRegister.from_dict(
            data.get("catalogs", dict()))
        # job list
        self.jobs: list = setup.get("jobs", [])

    @classmethod
    def create(
        cls,
        setupfile: Path | str,
        config: Configuration,
        cachepath: Path | str | None = None,
        backend: str = "scipy"
    ) -> Setup:
        new = cls.__new__(cls)
        new._path = setupfile
        if new.path.exists():
            raise FileExistsError(f"setup file '{new.path}' already exists")
        new.backend_name = backend
        # config can be copied
        new.config = config
        # set up the data management
        if cachepath is None:
            new.cache = None
        else:
            new.cache = CacheDirectory(cachepath)
        new.catalogs = InputRegister()
        # job list
        new.jobs = []
        new.write()
        new.reload()
        return new

    @property
    def path(self) -> Path:
        return Path(self._path)

    def write(self) -> None:
        setup = dict(
            configuration=self.config.to_dict(),
            data=dict(
                cachepath=str(self.cache) if self.cache is not None else None,
                catalogs=self.catalogs.to_dict()),
            backend=self.backend_name)
        if len(self.jobs) > 0:
            setup["jobs"] = self.jobs
        string = yaml.dump(setup)
        with open(self.path, "w") as f:
            f.write(string)

    @property
    def backend(self) -> ModuleType:
        if self._backend is None:
            try:
                self._backend = importlib.import_module(
                    f"yaw.{self.backend_name}")
            except ImportError as e:
                raise InvalidBackendError(
                    f"backend '{self.backend_name}' invalid or "
                    "yaw import failed") from e
        return self._backend

    @property
    def cache_restore_supported(self) -> bool:
        return hasattr(self.backend.Catalog, "from_cache")

    def set_reference(
        self,
        data: Input,
        rand: Input | None = None
    ) -> None:
        self.catalogs.set_reference(data, rand)
    
    def add_unknown(
        self,
        bin_idx: int,
        data: Input,
        rand: Input | None = None
    ) -> None:
        self.catalogs.add_unknown(bin_idx, data, rand)

    def _load_catalog(
        self,
        sample: str,
        kind: str,
        bin_idx: int | None = None
    ) -> CatalogBase:
        # get the correct sample type
        if sample == "reference":
            inputs = self.catalogs.get_reference()
        elif sample == "unknown":
            if bin_idx is None:
                raise ValueError("no 'bin_idx' provided")
            inputs = self.catalogs.get_unknown(bin_idx)
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
        kwargs.pop("index", None)
        kwargs.pop("cache", False)
        # attempt to load the catalog into memory
        if self.cache is None:
            # no cache available
            if input.cache:
                warnings.warn(
                    "no cache directory provided, cannot cache catalog")
            catalog = self.backend.Catalog.from_file(**kwargs)

        else:
            # cache available, get object path
            if sample == "reference":
                cachepath = self.cache.get_reference()[kind]
            else:
                cachepath = self.cache.get_unknown(bin_idx)[kind]
            exists = cachepath.exists()
            if self.cache_restore_supported and exists and input.cache:
                # already cached and backend supports restoring
                catalog = self.backend.Catalog.from_cache(cachepath)
            else:
                # no caching requested or not supported
                if input.cache:
                    kwargs["cache_directory"] = str(cachepath)
                    cachepath.mkdir(exist_ok=True)
                catalog = self.backend.Catalog.from_file(**kwargs)
        return catalog

    def load_reference(self, kind: str) -> CatalogBase:
        return self._load_catalog("reference", kind)

    def load_unknown(self, kind: str, bin_idx: int) -> dict[str, Path]:
        return self._load_catalog("unknown", kind, bin_idx=bin_idx)

    def list_catalogs(self) -> None:
        print(yaml.dump(self.catalogs.to_dict()))

    def add_job(self, job) -> None:
        self.jobs.append(job)

    def iter_jobs(self) -> Iterator:
        for job in self.jobs:
            yield job


def _setup_path(path: Path) -> Path:
    return path.joinpath("setup.yaml")


class ProjectDirectory:

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(
                f"project directory '{self.path}' does not exist")
        self._setup = Setup(_setup_path(self.path))
        # create any missing directories
        self.counts_dir.mkdir(exist_ok=True)
        self.estimate_dir.mkdir(exist_ok=True)

    @classmethod
    def create(
        cls,
        path: Path | str,
        config: Configuration,
        cachepath: Path | str | None = None,
        backend: str = "scipy"
    ) -> ProjectDirectory:
        path = Path(path).expanduser()
        path.mkdir(parents=True, exist_ok=False)
        if cachepath is None:
            cachepath = path.joinpath("cache")
        Setup.create(
            _setup_path(path), config=config,
            cachepath=cachepath, backend=backend)
        return cls(path)

    def __enter__(self) -> ProjectDirectory:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            self.setup.write()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def counts_dir(self) -> CountsDirectory:
        return CountsDirectory(self.path.joinpath("paircounts"))

    @property
    def estimate_dir(self) -> EstimateDirectory:
        return self.path.joinpath("estimate")

    def list_estimate_scales(self) -> list(str):
        return [path.name for path in self.estimate_dir.iterdir()]

    def get_estimate(self, scale_key: str) -> EstimateDirectory:
        path = self.estimate_dir.joinpath(scale_key)
        if not path.exists():
            raise ValueError(f"estimate for scale '{scale_key}' does not exist")
        return EstimateDirectory(path)
