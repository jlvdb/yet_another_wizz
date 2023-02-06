from __future__ import annotations

import importlib
import warnings
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


def _write_setup_file(
    path: Path | str,
    config: Configuration,
    cache_dir: Path | str | None,
    catalogs: InputRegister,
    backend_name: str,
) -> None:
    setup = dict(
        configuration=config.to_dict(),
        data=dict(
            cachepath=str(cache_dir) if cache_dir is not None else None,
            catalogs=catalogs.to_dict()),
        backend=backend_name)
    string = yaml.dump(setup)
    with open(path, "w") as f:
        f.write(string)


class ProjectDirectory:

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(
                f"project directory '{self.path}' does not exist")
        if not self.setup_file.exists:
            raise FileNotFoundError(
                f"setup file '{self.setup_file}' does not exist")
        self.setup_reload()
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
        new = cls.__new__(cls)  # access to path attributes
        new._path = Path(path).expanduser()
        new._path.mkdir(parents=True, exist_ok=False)
        if cachepath is None:
            cachepath = new._path.joinpath("cache")
        # create the setup file
        _write_setup_file(
            new.setup_file,
            config=config,
            cache_dir=cachepath,
            catalogs=InputRegister(),
            backend_name=backend)
        return cls(path)

    def __enter__(self) -> ProjectDirectory:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            self.setup_write()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("setup.yaml")

    def setup_reload(self) -> None:
        with open(self.setup_file) as f:
            setup = yaml.safe_load(f.read())
        # get the backend first which is used to import the backend package
        self._backend_name = setup["backend"]
        self._backend = None
        self.backend  # try to import
        # configuration is straight forward
        try:
            self._config = Configuration.from_dict(setup["configuration"])
        except KeyError as e:
            _parse_section_error(e, "configuration")
        # set up the data management
        try:
            data = setup["data"]
        except KeyError as e:
            _parse_section_error(e, "data")
        cachepath = data.get("cachepath")
        self._cache = CacheDirectory(cachepath)
        self._cache.mkdir(exist_ok=True, parents=True)
        self._inputs = InputRegister.from_dict(
            data.get("catalogs", dict()))

    def setup_write(self) -> None:
        _write_setup_file(
            self.setup_file,
            config=self.config,
            cache_dir=self.cache_dir,
            catalogs=self._inputs,
            backend_name=self._backend_name)

    @property
    def backend(self) -> ModuleType:
        if self._backend is None:
            try:
                self._backend = importlib.import_module(
                    f"yaw.{self._backend_name}")
            except ImportError as e:
                raise InvalidBackendError(
                    f"backend '{self._backend_name}' invalid or "
                    "yaw import failed") from e
        return self._backend

    @property
    def config(self) -> Configuration:
        return self._config

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

    def _load_catalog(
        self,
        sample: str,
        kind: str,
        bin_idx: int | None = None
    ) -> CatalogBase:
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
        if self.get_cache() is None:
            # no cache available
            if input.cache:
                warnings.warn(
                    "no cache directory provided, cannot cache catalog")
            catalog = self.backend.Catalog.from_file(**kwargs)

        else:
            # cache available, get object path
            if sample == "reference":
                cachepath = self._cache.get_reference()[kind]
            else:
                cachepath = self._cache.get_unknown(bin_idx)[kind]
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

    def load_unknown(self, kind: str, bin_idx: int) -> CatalogBase:
        return self._load_catalog("unknown", kind, bin_idx=bin_idx)

    def get_bin_indices(self) -> set[int]:
        return self._inputs.get_bin_indices()

    def show_catalogs(self) -> None:
        print(yaml.dump(self._inputs.to_dict()))

    @property
    def cache_dir(self) -> Path:
        return Path(self._cache)

    @property
    def cache_restore_supported(self) -> bool:
        return hasattr(self.backend.Catalog, "from_cache")

    def get_cache(self) -> CacheDirectory:
        return self._cache

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
        return [path.name for path in self.counts_dir.iterdir()]

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
        return [path.name for path in self.estimate_dir.iterdir()]
