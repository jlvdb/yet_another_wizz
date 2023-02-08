from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, get_args

import astropy.cosmology
import numpy as np
import yaml

from yaw import __version__
from yaw.core.cosmology import (
    TypeCosmology, get_default_cosmology, r_kpc_to_angle)
from yaw.core.utils import scales_to_keys

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from yaw.core.catalog import CatalogBase


logger = logging.getLogger(__name__.replace(".core.", "."))


class ConfigurationError(Exception):
    pass


def _parse_section_error(exception: Exception, section: str) -> NoReturn:
    msg = exception.args[0]
    item = msg.split("'")[1]
    if isinstance(exception, TypeError):
        if "__init__() got an unexpected keyword argument" in msg:
            raise ConfigurationError(
                f"encountered unknown option '{item}' in section '{section}'"
            ) from exception
        elif "missing" in msg:
            raise ConfigurationError(
                f"missing option '{item}' in section '{section}'"
            ) from exception
    elif isinstance(exception, KeyError):
        raise ConfigurationError(f"missing section '{section}'") from exception
    raise


def _array_equal(arr1: NDArray, arr2: NDArray) -> bool:
    return (
        isinstance(arr1, np.ndarray) and
        isinstance(arr2, np.ndarray) and
        arr1.shape == arr2.shape and
        (arr1 == arr2).all())


def _check_version(version: str) -> None:
    msg = "configuration has be generated on a new version than installed: "
    msg += f"{version} > {__version__}"
    this = [int(s) for s in __version__.split(".")]
    other = [int(s) for s in version.split(".")]
    for t, o in zip(this, other):
        if t > o:
            break
        elif t < o:
            raise ConfigurationError(msg)
    else:
        # check if this is a subversion
        if len(other) > len(this):
            raise ConfigurationError(msg)


def _cosmology_to_yaml(cosmology: TypeCosmology) -> str:
    if not isinstance(cosmology, astropy.cosmology.FLRW):
        raise ConfigurationError("cannot serialise custom cosmoligies to YAML")
    if cosmology.name not in astropy.cosmology.available:
        raise ConfigurationError(
            "can only serialise predefined astropy cosmologies to YAML")
    return cosmology.name


def _yaml_to_cosmology(cosmo_name: str) -> TypeCosmology:
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigurationError(
            f"unknown cosmology with name '{cosmo_name}', see "
            "'astropy.cosmology.available'")
    return getattr(astropy.cosmology, cosmo_name)


def _parse_cosmology(cosmology: TypeCosmology | str | None) -> TypeCosmology:
    if cosmology is None:
        cosmology = get_default_cosmology()
    elif isinstance(cosmology, str):
        cosmology = _yaml_to_cosmology(cosmology)
    elif not isinstance(cosmology, get_args(TypeCosmology)):
        which = ", ".join(get_args(TypeCosmology))
        raise ConfigurationError(
            f"'cosmology' must be instance of: {which}")
    return cosmology


@dataclass(frozen=True)
class ScalesConfig:

    rmin: Sequence[float] | float
    rmax: Sequence[float] | float
    rweight: float | None = field(default=None)
    rbin_num: int = field(default=50)

    def __post_init__(self) -> None:
        msg_scale_error = f"scales violates 'rmin' < 'rmax'"
        # validation, set to basic python types
        scalars = (float, int, np.number)
        if isinstance(self.rmin, Sequence) and isinstance(self.rmax, Sequence):
            if len(self.rmin) != len(self.rmax):
                raise ConfigurationError(
                    "number of elements in 'rmin' and 'rmax' do not match")
            # for clean YAML conversion
            for rmin, rmax in zip(self.rmin, self.rmax):
                if rmin >= rmax:
                    raise ConfigurationError(msg_scale_error)
            if len(self.rmin) == 1:
                rmin = float(self.rmin[0])
                rmax = float(self.rmax[0])
            else:
                rmin = list(float(f) for f in self.rmin)
                rmax = list(float(f) for f in self.rmax)
            object.__setattr__(self, "rmin", rmin)
            object.__setattr__(self, "rmax", rmax)
        elif isinstance(self.rmin, scalars) and isinstance(self.rmax, scalars):
            # for clean YAML conversion
            object.__setattr__(self, "rmin", float(self.rmin))
            object.__setattr__(self, "rmax", float(self.rmax))
            if self.rmin >= self.rmax:
                raise ConfigurationError(msg_scale_error)
        else:
            raise ConfigurationError(
                "'rmin' and 'rmax' must be both sequences or float")

    def __eq__(self, other: ScalesConfig) -> bool:
        if not _array_equal(self.scales, other.scales):
            return False
        if self.rweight != other.rweight:
            return False
        if self.rbin_num != other.rbin_num:
            return False
        return True

    @property
    def scales(self) -> NDArray[np.float_]:
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))

    def dict_keys(self) -> list[str]:
        return scales_to_keys(self.scales)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_manual_binning(
    zbins,
    *auto_args,
    require: bool = True,
    warn: bool = True
) -> bool:
    has_no_auto_args = any(val is None for val in auto_args)
    if zbins is None:
        if has_no_auto_args and require:
            raise ConfigurationError(
                "either 'zbins' or 'zmin', 'zmax', 'nbins' are required")
        return False
    else:
        if not has_no_auto_args and warn:
            logger.warn(
                "'zmin', 'zmax', 'nbins' are ignored if 'zbins' is provided")
        return True


class BinFactory:

    def __init__(
        self,
        zmin: float,
        zmax: float,
        nbins: int,
        cosmology: TypeCosmology | None = None
    ):
        if zmin >= zmax:
            raise ValueError("'zmin' >= 'zmax'")
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins

    def linear(self) -> NDArray[np.float_]:
        return np.linspace(self.zmin, self.zmax, self.nbins + 1)

    def comoving(self) -> NDArray[np.float_]:
        cbinning = np.linspace(
            self.cosmology.comoving_distance(self.zmin).value,
            self.cosmology.comoving_distance(self.zmax).value,
            self.nbins + 1)
        # construct a spline mapping from comoving distance to redshift
        zarray = np.linspace(0, 10.0, 5000)
        carray = self.cosmology.comoving_distance(zarray).value
        return np.interp(cbinning, xp=carray, fp=zarray)  # redshift @ cbinning

    def logspace(self) -> NDArray[np.float_]:
        logbinning = np.linspace(
            np.log(1.0 + self.zmin), np.log(1.0 + self.zmax), self.nbins + 1)
        return np.exp(logbinning) - 1.0

    @staticmethod
    def check(zbins: NDArray[np.float_]) -> None:
        if np.any(np.diff(zbins) <= 0):
            raise ValueError("redshift bins must be monotonicaly increasing")

    def get(self, method: str) -> NDArray[np.float_]:
        try:
            return getattr(self, method)()
        except AttributeError as e:
            raise ValueError(f"invalid binning method '{method}'") from e


@dataclass(frozen=True)
class BaseBinningConfig:

    zbins: NDArray[np.float_]

    @property
    def zmin(self) -> float:
        return float(self.zbins[0])

    @property
    def zmax(self) -> float:
        return float(self.zbins[-1])

    @property
    def zbin_num(self) -> int:
        return len(self.zbins) - 1


@dataclass(frozen=True)
class ManualBinningConfig(BaseBinningConfig):

    zbins: NDArray[np.float_]

    def __post_init__(self) -> None:
        if len(self.zbins) < 2:
            raise ConfigurationError("'zbins' must have at least two edges")
        BinFactory.check(self.zbins)
        object.__setattr__(self, "zbins", np.asarray(self.zbins))

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not _array_equal(self.zbins, other.zbins):
            return False
        return True

    @property
    def method(self) -> str:
        return "manual"

    def to_dict(self) -> dict[str, Any]:
        return dict(method=self.method, zbins=self.zbins.tolist())


@dataclass(frozen=True)
class AutoBinningConfig(BaseBinningConfig):

    zbins: NDArray[np.float_]
    method: str

    @classmethod
    def generate(
        cls,
        zmin: float,
        zmax: float,
        zbin_num: int,
        method: str = "linear",
        cosmology: TypeCosmology | None = None
    ) -> ScalesConfig:
        zbins = BinFactory(zmin, zmax, zbin_num, cosmology).get(method)
        return cls(zbins, method)

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not _array_equal(self.zbins, other.zbins):
            return False
        if self.method != other.method:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return dict(
            zmin=self.zmin,
            zmax=self.zmax,
            zbin_num=self.zbin_num,
            method=self.method)


@dataclass(frozen=True)
class BackendConfig:

    # general
    thread_num: int | None = field(default=None)
    # scipy
    crosspatch: bool = field(default=True)
    # treecorr
    rbin_slop: float = field(default=0.01)

    def __post_init__(self) -> None:
        if self.thread_num is None:
            object.__setattr__(self, "thread_num", os.cpu_count())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Configuration:

    scales: ScalesConfig
    binning: AutoBinningConfig | ManualBinningConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    cosmology: TypeCosmology | str | None = field(default=None)

    def __post_init__(self) -> None:
        # parse cosmology
        if self.cosmology is None:
            cosmology = get_default_cosmology()
        elif isinstance(self.cosmology, str):
            cosmology = _yaml_to_cosmology(self.cosmology)
        elif not isinstance(self.cosmology, get_args(TypeCosmology)):
            which = ", ".join(get_args(TypeCosmology))
            raise ConfigurationError(
                f"'cosmology' must be instance of: {which}")
        else:
            cosmology = self.cosmology
        cosmology = _parse_cosmology(self.cosmology)
        object.__setattr__(self, "cosmology", cosmology)

    @classmethod
    def create(
        cls,
        *,
        cosmology: TypeCosmology | str | None = None,
        # ScalesConfig
        rmin: ArrayLike,
        rmax: ArrayLike,
        rweight: float | None = None,
        rbin_num: int = 50,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: float | None = None,
        zmax: float | None = None,
        zbin_num: int | None = None,
        method: str = "linear",
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = None,
        crosspatch: bool = True,
        rbin_slop: float = 0.01
    ) -> Configuration:
        cosmology = _parse_cosmology(cosmology)
        scales = ScalesConfig(
            rmin=rmin, rmax=rmax, rweight=rweight, rbin_num=rbin_num)
        if _is_manual_binning(zbins, zmin, zmax, zbin_num):
            binning = ManualBinningConfig(zbins)
        else:
            binning = AutoBinningConfig.generate(
                zmin=zmin, zmax=zmax, zbin_num=zbin_num, method=method,
                cosmology=cosmology)
        backend = BackendConfig(
            thread_num=thread_num, crosspatch=crosspatch, rbin_slop=rbin_slop)
        return cls(
            scales=scales, binning=binning,
            backend=backend, cosmology=cosmology)

    def modify(
        self,
        *,
        cosmology: TypeCosmology | str | None = None,
        # ScalesConfig
        rmin: ArrayLike | None = None,
        rmax: ArrayLike | None = None,
        rweight: float | None = None,
        rbin_num: int | None = None,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: float | None = None,
        zmax: float | None = None,
        zbin_num: int | None = None,
        method: str | None = None,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = None,
        crosspatch: bool | None = None,
        rbin_slop: float | None = None
    ) -> Configuration:
        config = self.to_dict()
        if cosmology is not None:
            if isinstance(cosmology, str):
                cosmology = _yaml_to_cosmology(cosmology)
            config["cosmology"] = _cosmology_to_yaml(cosmology)
        # ScalesConfig
        if rmin is not None:
            config["scales"]["rmin"] = rmin
        if rmax is not None:
            config["scales"]["rmax"] = rmax
        if rweight is not None:
            config["scales"]["rweight"] = rweight
        if rbin_num is not None:
            config["scales"]["rbin_num"] = rbin_num
        # AutoBinningConfig /  ManualBinningConfig
        if _is_manual_binning(zbins, zmin, zmax, zbin_num, require=False):
            if zbins is not None:
                config["binning"]["zbins"] = zbins
        else:
            if zmin is not None:
                config["binning"]["zmin"] = zmin
            if zmax is not None:
                config["binning"]["zmax"] = zmax
            if zbin_num is not None:
                config["binning"]["zbin_num"] = zbin_num
            if method is not None:
                config["binning"]["method"] = method
        # BackendConfig
        if thread_num is not None:
            config["backend"]["thread_num"] = thread_num
        if crosspatch is not None:
            config["backend"]["crosspatch"] = crosspatch
        if rbin_slop is not None:
            config["backend"]["rbin_slop"] = rbin_slop
        return self.__class__.from_dict(config)

    def plot_scales(
        self,
        catalog: CatalogBase,
        log: bool = True,
        legend: bool = True
    ) -> Figure:
        import matplotlib.pyplot as plt

        fig, ax_scale = plt.subplots(1, 1)
        # plot scale of annulus
        for r_min, r_max in self.scales.scales:
            ang_min, ang_max = np.transpose([
                r_kpc_to_angle([r_min, r_max], z, self.cosmology)
                for z in self.binning.zbins])
            ax_scale.fill_between(
                self.binning.zbins, ang_min, ang_max, step="post", alpha=0.3,
                label=rf"${r_min:.0f} < r \leq {r_max:.0f}$ kpc")
        if legend:
            ax_scale.legend(loc="lower right")
        # plot patch sizes
        ax_patch = ax_scale.twiny()
        bins = np.histogram_bin_edges(catalog.radii)
        if log:
            ax_patch.set_yscale("log")
            bins = np.logspace(
                np.log10(bins[0]), np.log10(bins[-1]), len(bins), base=10.0)
        ax_patch.hist(
            catalog.radii, bins,
            orientation="horizontal", color="k", alpha=0.5)
        # decorate
        ax_scale.set_xlim(self.binning.zmin, self.binning.zmax)
        ax_scale.set_ylabel("Radius / rad")
        ax_scale.set_xlabel("Redshift")
        ax_patch.set_xlabel("Patch count")
        return fig

    @classmethod
    def from_dict(cls, config: dict) -> Configuration:
        _check_version(config.pop("version", "0.0"))
        cosmology = _parse_cosmology(config.pop("cosmology", None))
        # parse the required subgroups
        try:
            scales = ScalesConfig(**config.pop("scales"))
        except (TypeError, KeyError) as e:
            _parse_section_error(e, "scales")
        try:
            binning_conf = config.pop("binning")
            if "zbins" in binning_conf:
                binning = ManualBinningConfig(binning_conf["zbins"])
            else:
                binning = AutoBinningConfig.generate(
                    cosmology=cosmology, **binning_conf)
        except (TypeError, KeyError) as e:
            _parse_section_error(e, "binning")
        # parse the optional subgroups
        try:
            backend = BackendConfig(**config.pop("backend"))
        except KeyError:
            backend = BackendConfig()
        except TypeError as e:
            _parse_section_error(e, "backend")
        # check that there are no entries left
        if len(config) > 0:
            key = next(iter(config.keys()))
            raise ConfigurationError(f"encountered unknown section '{key}'")
        return cls(
            scales=scales, binning=binning,
            backend=backend, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        values = dict(version=__version__)
        for attr in asdict(self):
            value = getattr(self, attr)  # avoid asdict() recursion
            if attr == "cosmology":
                values[attr] = _cosmology_to_yaml(value)
            else:
                values[attr] = value.to_dict()
        return values

    @classmethod
    def from_yaml(cls, path: str) -> Configuration:
        logger.info(f"reading configuration file '{path}'")
        with open(path) as f:
            config = yaml.safe_load(f.read())
        return cls.from_dict(config)

    def to_yaml(self, path: str) -> None:
        logger.info(f"writing configuration file '{path}'")
        string = yaml.dump(self.to_dict())
        with open(path, "w") as f:
            f.write(string)
