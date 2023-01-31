from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NoReturn

import yaml

from yet_another_wizz.core.config import Configuration

from yet_another_wizz.infrastructure.data import InputRegister
from yet_another_wizz.infrastructure.directories import (
    CountsDirectory, EstimateDirectory)

if TYPE_CHECKING:
    from yet_another_wizz.core.catalog import CatalogBase
    from yet_another_wizz.infrastructure.data import Input


class InvalidBackendError(Exception):
    pass


class SetupError(Exception):
    pass


def _parse_section_error(exception: Exception, section: str) -> NoReturn:
    if isinstance(exception, KeyError):
        raise SetupError(f"missing section '{section}'")
    raise


@dataclass
class DataCache:
    path: Path | None
    restoring_support: bool

    def __post_init__(self) -> None:
        if self.defined:
            self.path = Path(self.path)
            self.path.mkdir(exist_ok=True, parents=True)

    @property
    def defined(self) -> bool:
        return self.path is not None



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
        self.cache = DataCache(
            path=data.get("cachepath"),
            restoring_support=hasattr(self.backend.Catalog, "from_cache"))
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
        new.cachepath = cachepath
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
                cachepath=str(self.cache.path) if self.cache.defined else None,
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
                    f"yet_another_wizz.{self.backend_name}")
            except ImportError:
                raise InvalidBackendError(
                    f"backend '{self.backend_name}' does not exist, "
                    "yet_another_wizz import failed")
        return self._backend

    def add_catalog(
        self,
        identifier: str,
        entry: Input,
        force: bool = False
    ) -> None:
        self.catalogs.add(identifier, entry, force)

    def load_catalog(self, identifier: str) -> CatalogBase:
        entry = self.catalogs.get(identifier)
        # get arguments for Catalog.from_file()
        kwargs = entry.to_dict()
        kwargs.pop("index", None)
        kwargs.pop("cache", None)

        # perform file operations
        if not self.cache.defined:
            # no cache available
            if entry.cache:
                warnings.warn(
                    "no cache directory provided, cannot cache catalog")
            catalog = self.backend.Catalog.from_file(**kwargs)

        else:
            # cache available
            cache_entry = self.cache.path.joinpath(identifier)
            exists = cache_entry.exists()
            if self.cache.restoring_support and exists and entry.cache:
                # already cached and backend supports restoring
                catalog = self.backend.Catalog.from_cache(cache_entry)
            else:
                # no caching requested or not supported
                catalog = self.backend.Catalog.from_file(**kwargs)
        return catalog

    def list_catalogs(self) -> None:
        print(yaml.dump(self.catalogs.to_dict()))

    def add_job(self, job) -> None:
        self.jobs.append(job)

    def list_jobs(self) -> None:
        print(yaml.dump(self.jobs))


class ProjectDirectory:

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self._setup = Setup(self.setup_file)

    @classmethod
    def create(
        cls,
        path: Path | str,
        config: Configuration,
        cachepath: Path | str,
        backend: str = "scipy"
    ) -> ProjectDirectory:
        new = cls.__new__(cls)
        new.path = Path(path).expanduser()
        new.path.mkdir(parents=True, exist_ok=False)
        new._setup = Setup.create(
            new.setup_file, config=config, cachepath=cachepath, backend=backend)
        return new

    @property
    def setup_file(self) -> Path:
        return self.path.joinpath("setup.yaml")

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def cache_dir(self) -> Path:
        return self._setup.cachepath

    @property
    def counts_dir(self) -> Path:
        return self.path.joinpath()

    def list_counts_scales(self) -> list(str):
        return [path.name for path in self.counts_dir.iterdir()]

    def get_counts(self, scale_key: str) -> CountsDirectory:
        path = self.cache_dir.joinpath(scale_key)
        if not path.exists():
            raise ValueError(f"counts for scale '{scale_key}' do not exist")
        return CountsDirectory(path)

    @property
    def estimate_dir(self) -> EstimateDirectory:
        return self.path.joinpath("estimate")

    def list_estimate_scales(self) -> list(str):
        return [path.name for path in self.estimate_dir.iterdir()]

    def get_estimate(self, scale_key: str) -> EstimateDirectory:
        path = self.cache_dir.joinpath(scale_key)
        if not path.exists():
            raise ValueError(f"estimate for scale '{scale_key}' does not exist")
        return EstimateDirectory(path)
