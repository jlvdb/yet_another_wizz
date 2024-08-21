from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args

import astropy.cosmology
import numpy as np
import yaml
from numpy.typing import NDArray

from yaw.containers import Binning, RedshiftBinningFactory, Serialisable
from yaw.containers import Tbin_method as Tbin_method_auto
from yaw.containers import Tclosed, Tpath, default_bin_method
from yaw.cosmology import (
    CustomCosmology,
    Tcosmology,
    cosmology_is_equal,
    get_default_cosmology,
)

T = TypeVar("T")
Tyaml = TypeVar("Tyaml", bound="YamlSerialisable")
Tbase_config = TypeVar("Tbase_config", bound="BaseConfig")

Tbin_method_all = Union[Tbin_method_auto | Literal["manual"]]


class _NotSet_meta(type):
    def __repr__(self) -> str:
        return "NotSet"  # pragma: no cover

    def __bool__(self) -> bool:
        return False


class NotSet(metaclass=_NotSet_meta):
    pass


class ConfigError(Exception):
    pass


def cosmology_to_yaml(cosmology: Tcosmology) -> str:
    if isinstance(cosmology, CustomCosmology):
        raise ConfigError("cannot serialise custom cosmologies to YAML")

    elif not isinstance(cosmology, astropy.cosmology.FLRW):
        raise TypeError(f"invalid type '{type(cosmology)}' for cosmology")

    if cosmology.name not in astropy.cosmology.available:
        raise ConfigError("can only serialise predefined astropy cosmologies to YAML")

    return cosmology.name


def yaml_to_cosmology(cosmo_name: str) -> Tcosmology:
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigError(
            "unknown cosmology, for available options see 'astropy.cosmology.available'"
        )

    return getattr(astropy.cosmology, cosmo_name)


def parse_cosmology(cosmology: Tcosmology | str | None) -> Tcosmology:
    if cosmology is None:
        return get_default_cosmology()

    elif isinstance(cosmology, str):
        return yaml_to_cosmology(cosmology)

    elif not isinstance(cosmology, get_args(Tcosmology)):
        which = ", ".join(str(c) for c in get_args(Tcosmology))
        raise ConfigError(f"'cosmology' must be instance of: {which}")

    return cosmology


class YamlSerialisable(Serialisable):
    @classmethod
    def from_file(cls: type[Tyaml], path: Tpath) -> Tyaml:
        with Path(path).open() as f:
            kwarg_dict = yaml.safe_load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Tpath) -> None:
        with Path(path).open(mode="w") as f:
            yaml.safe_dump(self.to_dict(), f)


class Immutable:
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"attribute '{name}' is immutable")


class BaseConfig(YamlSerialisable, Immutable):
    @classmethod
    def from_dict(
        cls: type[Tbase_config],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any],  # passing additional constructor data
    ) -> Tbase_config:
        return cls.create(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    def create(cls: type[Tbase_config], **kwargs: Any) -> Tbase_config:
        return cls.from_dict(kwargs)

    @abstractmethod
    def modify(self: Tbase_config, **kwargs: Any | NotSet) -> Tbase_config:
        conf_dict = self.to_dict()
        conf_dict.update(
            {key: value for key, value in kwargs.items() if value is not NotSet}
        )
        return type(self).from_dict(conf_dict)

    @abstractmethod
    def __eq__(self) -> bool:
        pass


def parse_optional(value: Any | None, type: type[T]) -> T | None:
    if value is None:
        return None

    return type(value)


default_rweight = None
default_resolution = None


class ScalesConfig(BaseConfig):
    rmin: list[float] | float
    rmax: list[float] | float
    rweight: float | None
    resolution: int | None

    def __init__(
        self,
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        *,
        rweight: float | None = default_rweight,
        resolution: int | None = default_resolution,
    ) -> None:
        rmin: NDArray = np.atleast_1d(rmin)
        rmax: NDArray = np.atleast_1d(rmax)

        if rmin.ndim != rmax.ndim and rmin.ndim != 1:
            raise ConfigError("'rmin/rmax' must be scalars or one-dimensional arrays")
        if len(rmin) != len(rmax):
            raise ConfigError("number of elements in 'rmin' and 'rmax' does not match")

        if np.any(np.diff(rmin, rmax) <= 0.0):
            raise ConfigError("'rmin' must be smaller than corresponding 'rmax'")

        # ensure a YAML-friendly, i.e. native python, type
        object.__setattr__(self, "rmin", rmin.squeeze().tolist())
        object.__setattr__(self, "rmax", rmax.squeeze().tolist())
        object.__setattr__(self, "rweight", parse_optional(rweight, float))
        object.__setattr__(self, "resolution", parse_optional(resolution, int))

    @property
    def num_scales(self) -> int:
        try:
            return len(self.rmin)
        except TypeError:
            return 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            np.array_equal(self.rmin, other.rmin)
            and np.array_equal(self.rmax, other.rmax)
            and self.rweight == other.rweight
            and self.rbin_num == other.rbin_num
        )

    def to_dict(self) -> dict[str, Any]:
        attrs = ("rmin", "rmax", "rweight", "resolution")
        return {attr: getattr(self, attr) for attr in attrs}

    @classmethod
    def create(
        self,
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        rweight: float | None = default_rweight,
        resolution: int | None = default_resolution,
    ) -> ScalesConfig:
        return super().create(rmin, rmax, rweight, resolution)

    def modify(
        self,
        rmin: Iterable[float] | float = NotSet,
        rmax: Iterable[float] | float = NotSet,
        rweight: float | None = NotSet,
        resolution: int | None = NotSet,
    ) -> ScalesConfig:
        return super().modify(rmin, rmax, rweight, resolution)


default_num_bins = 30


class BinningConfig(BaseConfig):
    binning: Binning
    method: Tbin_method_all

    def __init__(
        self,
        binning: Binning,
        method: Tbin_method_all = default_bin_method,
    ) -> None:
        if not isinstance(binning, Binning):
            raise TypeError(f"'binning' must be of type '{type(binning)}'")
        object.__setattr__(self, "binning", binning)

        method = parse_optional(method, str)
        if method not in get_args(Tbin_method_auto) and method != "manual":
            raise ValueError(f"invalid binning method '{method}'")
        object.__setattr__(self, "method", method)

    @property
    def zmin(self) -> float:
        return float(self.binning.edges[0])

    @property
    def zmax(self) -> float:
        return float(self.binning.edges[-1])

    @property
    def num_bins(self) -> int:
        return len(self.binning)

    @property
    def closed(self) -> Tbin_method_all:
        return self.binning.closed

    @property
    def is_manual(self) -> bool:
        return self.method == "manual"

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, Any], cosmology: Tcosmology | None = None
    ) -> BinningConfig:
        if the_dict["method"] == "manual":
            edges = the_dict.pop("edges")
            closed = the_dict.pop("closed")
            binning = Binning(edges, closed=closed)
            return cls(binning, **the_dict)

        return cls.create(**the_dict, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        if self.is_manual:
            the_dict = dict(method=self.method, edges=self.binning.edges)

        else:
            the_dict = dict(
                zmin=self.zmin,
                zmax=self.zmax,
                num_bins=self.num_bins,
                method=self.method,
            )

        the_dict["closed"] = self.closed
        return the_dict

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.method == other.method and self.binning == other.binning

    @classmethod
    def create(
        cls,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
        num_bins: int = default_num_bins,
        method: Tbin_method_all = default_bin_method,
        edges: Iterable[float] | None = None,
        closed: Tclosed | None = None,
        cosmology: Tcosmology | str | None = None,
    ) -> BinningConfig:
        auto_args_set = (zmin is not None, zmax is not None)
        manual_args_set = (edges is not None,)
        if not all(manual_args_set) and not all(auto_args_set):
            raise ConfigError("either 'edges' or 'zmin' and 'zmax' are required")

        elif all(auto_args_set):  # generate bin edges
            if all(manual_args_set):
                warnings.warn(
                    "'zbins' set but ignored since 'zmin' and 'zmax' are provided"
                )
            bin_func = RedshiftBinningFactory(cosmology).get_method(method)
            edges = bin_func(zmin, zmax, num_bins, closed=closed)

        else:  # use provided bin edges
            method = "manual"

        binning = Binning(edges, closed=closed)
        return cls(binning, method=method)

    def modify(
        self,
        *,
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = default_num_bins,
        method: Tbin_method_all | NotSet = NotSet,
        edges: Iterable[float] | NotSet = NotSet,
        closed: Tclosed | NotSet = NotSet,
        cosmology: Tcosmology | str | None | NotSet = NotSet,
    ) -> BinningConfig:
        if edges is NotSet:
            if method == "manual":
                raise ConfigError("'method' is 'manual' but no bin edges provided")
            the_dict = dict()
            the_dict["zmin"] = self.zmin if zmin is NotSet else zmin
            the_dict["zmax"] = self.zmax if zmax is NotSet else zmax
            the_dict["num_bins"] = self.num_bins if num_bins is NotSet else num_bins
            the_dict["method"] = self.method if method is NotSet else method
            the_dict["closed"] = self.closed if closed is NotSet else closed

        else:
            the_dict = dict(edges=edges)
            the_dict["closed"] = self.closed if closed is NotSet else closed
            the_dict["method"] = "manual"

        return type(self).from_dict(the_dict, cosmology=cosmology)


default_cosmology = get_default_cosmology().name


class Configuration(BaseConfig):
    scales: ScalesConfig
    binning: BinningConfig
    cosmology: Tcosmology | str

    def __init__(
        self,
        scales: ScalesConfig,
        binning: BinningConfig,
        cosmology: Tcosmology | str | None = None,
    ) -> None:
        if not isinstance(scales, ScalesConfig):
            raise TypeError(f"'scales' must be of type '{type(ScalesConfig)}'")
        object.__setattr__(self, "scales", scales)

        if not isinstance(binning, BinningConfig):
            raise TypeError(f"'binning' must be of type '{type(BinningConfig)}'")
        object.__setattr__(self, "binning", binning)

        object.__setattr__(self, "cosmology", parse_cosmology(cosmology))

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> Configuration:
        the_dict = the_dict.copy()

        cosmology = parse_cosmology(the_dict.pop("cosmology", default_cosmology))

        try:  # scales
            scales_dict = the_dict.pop("scales")
            scales = ScalesConfig.from_dict(scales_dict)
        except (TypeError, KeyError) as err:
            raise ConfigError("failed parsing the 'scales' section") from err

        try:  # binning
            binning_dict = the_dict.pop("binning")
            binning = BinningConfig.from_dict(binning_dict, cosmology=cosmology)
        except (TypeError, KeyError) as err:
            raise ConfigError("failed parsing the 'binning' section") from err

        if len(the_dict) > 0:
            key = next(iter(the_dict))
            raise ConfigError(f"encountered unknown configuration entry '{key}'")

        return cls(scales=scales, binning=binning, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            scales=self.scales.to_dict(),
            binning=self.binning.to_dict(),
            cosmology=cosmology_to_yaml(self.cosmology),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            self.binning == other.binning
            and self.scales == other.scales
            and cosmology_is_equal(self.cosmology, other.cosmology)
        )

    @classmethod
    def create(
        cls,
        *,
        cosmology: Tcosmology | str | None = default_cosmology,
        # ScalesConfig
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        rweight: float | None = default_rweight,
        resolution: int | None = default_resolution,
        # BinningConfig
        zmin: float | None = None,
        zmax: float | None = None,
        num_bins: int = default_num_bins,
        method: Tbin_method_all = default_bin_method,
        edges: Iterable[float] | None = None,
        closed: Tclosed | None = None,
    ) -> Configuration:
        cosmology = parse_cosmology(cosmology)
        scales = ScalesConfig.create(
            rmin=rmin, rmax=rmax, rweight=rweight, resolution=resolution
        )
        binning = BinningConfig.create(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
            cosmology=cosmology,
        )
        return cls(scales=scales, binning=binning, cosmology=cosmology)

    def modify(
        self,
        *,
        cosmology: Tcosmology | str | None | NotSet = NotSet,
        # ScalesConfig
        rmin: Iterable[float] | float | NotSet = NotSet,
        rmax: Iterable[float] | float | NotSet = NotSet,
        rweight: float | None | NotSet = NotSet,
        resolution: int | None | NotSet = NotSet,
        # BinningConfig
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = NotSet,
        method: Tbin_method_all | NotSet = NotSet,
        edges: Iterable[float] | None = NotSet,
        closed: Tclosed | NotSet = NotSet,
    ) -> Configuration:
        cosmology = (
            self.cosmology if cosmology is NotSet else parse_cosmology(cosmology)
        )
        scales = self.scales.modify(
            rmin=rmin, rmax=rmax, rweight=rweight, resolution=resolution
        )
        binning = self.binning.modify(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
            cosmology=cosmology,
        )
        return type(self)(scales=scales, binning=binning, cosmology=cosmology)
