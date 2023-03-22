from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, get_args

import astropy.cosmology
import numpy as np
import yaml

from yaw import default as DEFAULT
from yaw.default import NotSet
from yaw.cosmology import (
    BINNING_OPTIONS, COSMOLOGY_OPTIONS, BinFactory, TypeCosmology,
    get_default_cosmology, r_kpc_to_angle)
from yaw.utils import DictRepresentation, Parameter, array_equal, scales_to_keys

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from yaw.catalogs import BaseCatalog


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    pass


def parse_section_error(
    exception: Exception,
    section: str,
    reraise: Exception = ConfigurationError
) -> NoReturn:
    msg = exception.args[0]
    item = msg.split("'")[1]
    if isinstance(exception, TypeError):
        if "__init__() got an unexpected keyword argument" in msg:
            raise reraise(
                f"encountered unknown option '{item}' in section '{section}'"
            ) from exception
        elif "missing" in msg:
            raise reraise(
                f"missing option '{item}' in section '{section}'"
            ) from exception
    elif isinstance(exception, KeyError):
        raise reraise(f"missing section '{section}'") from exception
    raise


def cosmology_to_yaml(cosmology: TypeCosmology) -> str:
    if not isinstance(cosmology, astropy.cosmology.FLRW):
        raise ConfigurationError("cannot serialise custom cosmoligies to YAML")
    if cosmology.name not in astropy.cosmology.available:
        raise ConfigurationError(
            "can only serialise predefined astropy cosmologies to YAML")
    return cosmology.name


def yaml_to_cosmology(cosmo_name: str) -> TypeCosmology:
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigurationError(
            f"unknown cosmology with name '{cosmo_name}', see "
            "'astropy.cosmology.available'")
    return getattr(astropy.cosmology, cosmo_name)


def parse_cosmology(cosmology: TypeCosmology | str | None) -> TypeCosmology:
    if cosmology is None:
        cosmology = get_default_cosmology()
    elif isinstance(cosmology, str):
        cosmology = yaml_to_cosmology(cosmology)
    elif not isinstance(cosmology, get_args(TypeCosmology)):
        which = ", ".join(get_args(TypeCosmology))
        raise ConfigurationError(
            f"'cosmology' must be instance of: {which}")
    return cosmology


@dataclass(frozen=True)
class ScalesConfig(DictRepresentation):

    rmin: Sequence[float] | float = field(
        metadata=Parameter(
            type=float, nargs="*", required=True,
            help="(list of) lower scale limit in kpc (pyhsical)"))
    rmax: Sequence[float] | float = field(
        metadata=Parameter(
            type=float, nargs="*", required=True,
            help="(list of) upper scale limit in kpc (pyhsical)"))
    rweight: float | None = field(
        default=DEFAULT.Scales.rweight,
        metadata=Parameter(
            type=float,
            help="weight galaxy pairs by their separation to power 'rweight'",
            default_text="(default: no weighting applied)"))
    rbin_num: int = field(
        default=DEFAULT.Scales.rbin_num,
        metadata=Parameter(
            type=int,
            help="number of bins in log r used (i.e. resolution) to compute "
                 "distance weights",
            default_text="(default: %(default)s)"))

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

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> ScalesConfig:
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __eq__(self, other: ScalesConfig) -> bool:
        if not array_equal(self.as_array(), other.as_array()):
            return False
        if self.rweight != other.rweight:
            return False
        if self.rbin_num != other.rbin_num:
            return False
        return True

    def as_array(self) -> NDArray[np.float_]:
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))

    def dict_keys(self) -> list[str]:
        return scales_to_keys(self.as_array())


class BaseBinningConfig(DictRepresentation):

    zbins: NDArray[np.float_]
    method: str

    def __repr__(self) -> str:
        name = self.__class__.__name__
        zbin_num = self.zbin_num
        z = f"{self.zmin:.3f}...{self.zmax:.3f}"
        method = self.method
        return f"{name}({zbin_num=}, {z=}, {method=})"


@dataclass(frozen=True, repr=False)
class ManualBinningConfig(BaseBinningConfig):

    zbins: NDArray[np.float_] = field(
        metadata=Parameter(
            type=float, nargs="*",
            help="list of custom redshift bin edges, if provided, other "
                 "binning parameters are omitted, method is set to 'manual'"))

    def __post_init__(self) -> None:
        if len(self.zbins) < 2:
            raise ConfigurationError("'zbins' must have at least two edges")
        BinFactory.check(self.zbins)
        object.__setattr__(self, "zbins", np.asarray(self.zbins))

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not array_equal(self.zbins, other.zbins):
            return False
        return True

    @property
    def method(self) -> str:
        return "manual"

    @property
    def zmin(self) -> float:
        return float(self.zbins[0])

    @property
    def zmax(self) -> float:
        return float(self.zbins[-1])

    @property
    def zbin_num(self) -> int:
        return len(self.zbins) - 1

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        **kwargs
    ) -> ManualBinningConfig:
        return cls(np.asarray(the_dict["zbins"]))

    def to_dict(self) -> dict[str, Any]:
        return dict(method=self.method, zbins=self.zbins.tolist())


@dataclass(frozen=True, repr=False)
class AutoBinningConfig(BaseBinningConfig):

    zbins: NDArray[np.float_]
    method: str = field(
        default=DEFAULT.Configuration.binning.method,
        metadata=Parameter(
            choices=BINNING_OPTIONS,
            help="redshift binning method, 'logspace' means equal size in "
                 "log(1+z)",
            default_text="(default: %(default)s)"))
    zmin: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="lower redshift limit",
            default_text="(default: %(default)s)"))
    zmax: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="upper redshift limit",
            default_text="(default: %(default)s)"))
    zbin_num: int = field(
        default=DEFAULT.AutoBinning.zbin_num,
        init=False,
        metadata=Parameter(
            type=int,
            help="number of redshift bins",
            default_text="(default: %(default)s)"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "zmin", float(self.zbins[0]))
        object.__setattr__(self, "zmax", float(self.zbins[-1]))
        object.__setattr__(self, "zbin_num", len(self.zbins)-1)

    @classmethod
    def generate(
        cls,
        zmin: float,
        zmax: float,
        zbin_num: int = DEFAULT.AutoBinning.zbin_num,
        method: str = DEFAULT.AutoBinning.method,
        cosmology: TypeCosmology | None = None
    ) -> ScalesConfig:
        zbins = BinFactory(zmin, zmax, zbin_num, cosmology).get(method)
        return cls(zbins, method)

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not array_equal(self.zbins, other.zbins):
            return False
        if self.method != other.method:
            return False
        return True

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        cosmology: TypeCosmology | None = None
    ) -> AutoBinningConfig:
        return cls.generate(**the_dict, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            zmin=self.zmin,
            zmax=self.zmax,
            zbin_num=self.zbin_num,
            method=self.method)


def warn_binning_args_ignored(
    zmin: bool | float | None,
    zmax: bool | float | None,
    zbin_num: bool | int | None
) -> None:
    # NOTE: NotSet is also False
    if zmin or zmax or zbin_num:
        logger.warn(
            "'zmin', 'zmax', 'nbins' are ignored if 'zbins' is provided")


def make_binning_config(
    cosmology: TypeCosmology | str | None,
    zmin: float | None = None,
    zmax: float | None = None,
    zbin_num: int | None = None,
    method: str | None = None,
    zbins: NDArray[np.float_] | None = None,
) -> ManualBinningConfig | AutoBinningConfig:
    auto_args_set =  (zmin is not None, zmax is not None, zbin_num is not None)
    if zbins is None and not all(auto_args_set):
        raise ConfigurationError(
            "either 'zbins' or 'zmin', 'zmax', 'zbin_num' are required")
    elif all(auto_args_set):
        return AutoBinningConfig.generate(
            zmin=zmin, zmax=zmax, zbin_num=zbin_num,
            method=method, cosmology=cosmology)
    else:
        warn_binning_args_ignored(*auto_args_set)
        return ManualBinningConfig(zbins)


@dataclass(frozen=True)
class BackendConfig(DictRepresentation):

    # general
    thread_num: int | None = field(
        default=DEFAULT.Backend.thread_num,
        metadata=Parameter(
            type=int,
            help="default number of threads to use",
            default_text="(default: all)"))
    # scipy
    crosspatch: bool = field(
        default=DEFAULT.Backend.crosspatch,
        metadata=Parameter(
            type=bool,
            help="whether to count pairs across patch boundaries (scipy "
                 "backend only)"))
    # treecorr
    rbin_slop: float = field(
        default=DEFAULT.Backend.rbin_slop,
        metadata=Parameter(
            type=float,
            help="TreeCorr 'rbin_slop' parameter",
            default_text="(default: %(default)s), without 'rweight' this just "
                         "a single radial bin, otherwise 'rbin_num'"))

    def __post_init__(self) -> None:
        if self.thread_num is None:
            object.__setattr__(self, "thread_num", os.cpu_count())

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> BackendConfig:
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def get_threads(self, max=None) -> int:
        if self.thread_num is None:
            thread_num = os.cpu_count()
        else:
            thread_num = self.thread_num
        if max is not None:
            if max < 1:
                raise ValueError("'max' must be positive")
            thread_num = min(max, thread_num)
        return thread_num


@dataclass(frozen=True)
class Configuration(DictRepresentation):
    """The central configration for correlation measurements.

    Construct with .create() method.
    """

    scales: ScalesConfig
    binning: AutoBinningConfig | ManualBinningConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    cosmology: TypeCosmology | str | None = field(
        default=DEFAULT.Configuration.cosmology,
        metadata=Parameter(
            type=str, choices=COSMOLOGY_OPTIONS,
            help="cosmological model used for distance calculations",
            default_text="(see astropy.cosmology, default: %(default)s)"))

    def __post_init__(self) -> None:
        # parse cosmology
        if self.cosmology is None:
            cosmology = get_default_cosmology()
        elif isinstance(self.cosmology, str):
            cosmology = yaml_to_cosmology(self.cosmology)
        elif not isinstance(self.cosmology, get_args(TypeCosmology)):
            which = ", ".join(get_args(TypeCosmology))
            raise ConfigurationError(
                f"'cosmology' must be instance of: {which}")
        else:
            cosmology = self.cosmology
        cosmology = parse_cosmology(self.cosmology)
        object.__setattr__(self, "cosmology", cosmology)

    @classmethod
    def create(
        cls,
        *,
        cosmology: TypeCosmology | str | None = DEFAULT.Configuration.cosmology,
        # ScalesConfig
        rmin: ArrayLike,
        rmax: ArrayLike,
        rweight: float | None = DEFAULT.Configuration.scales.rweight,
        rbin_num: int = DEFAULT.Configuration.scales.rbin_num,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: ArrayLike,
        zmax: ArrayLike,
        zbin_num: int | None = DEFAULT.Configuration.binning.zbin_num,
        method: str = DEFAULT.Configuration.binning.method,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = DEFAULT.Configuration.backend.thread_num,
        crosspatch: bool = DEFAULT.Configuration.backend.crosspatch,
        rbin_slop: float = DEFAULT.Configuration.backend.rbin_slop
    ) -> Configuration:
        """Create a new configuration object.

        Keyword Args:
            cosmology (optional):
                Named astropy cosmology used to compute distances. For options
                see :obj:`~yaw.cosmology.COSMOLOGY_OPTIONS`.
            rmin (:obj:`ArrayLike`):
                (List of) lower scale limit in kpc (pyhsical).
            rmax (:obj:`ArrayLike`):
                (List of) upper scale limit in kpc (pyhsical).
            rweight (float, optional):
                Weight galaxy pairs by their separation to power 'rweight'.
            rbin_num (int, optional):
                Number of bins in log r used (i.e. resolution) to compute
                distance weights.
            zmin (float):
                Lower redshift limit.
            zmax (float):
                Upper redshift limit.
            zbin_num (int, optional):
                Number of redshift bins
            method (str, optional):
                Redshift binning method, 'logspace' means equal size in
                log(1+z).
            zbins (:obj:`NDArray`, optional):
                Manually define redshift bin edges.
            thread_num (int, optional):
                Default number of threads to use.
            crosspatch (bool, optional):
                whether to count pairs across patch boundaries (scipy backend
                only)
            rbin_slop (float, optional):
                TreeCorr 'rbin_slop' parameter
    
        Returns:
            :obj:`Configuration`
        
        .. Note::
            If provided, ``zbins`` takes precedence over ``zmin``, ``zmax``,
            and ``zbin_num``; ``method`` is automatically set to
            ``"manual"``.
        """
        cosmology = parse_cosmology(cosmology)
        scales = ScalesConfig(
            rmin=rmin, rmax=rmax, rweight=rweight, rbin_num=rbin_num)
        binning = make_binning_config(
            cosmology=cosmology, zmin=zmin, zmax=zmax, zbin_num=zbin_num,
            method=method, zbins=zbins)
        backend = BackendConfig(
            thread_num=thread_num, crosspatch=crosspatch, rbin_slop=rbin_slop)
        return cls(
            scales=scales, binning=binning,
            backend=backend, cosmology=cosmology)

    def modify(
        self,
        *,
        cosmology: TypeCosmology | str | None = NotSet,
        # ScalesConfig
        rmin: ArrayLike | None = NotSet,
        rmax: ArrayLike | None = NotSet,
        rweight: float | None = NotSet,
        rbin_num: int | None = NotSet,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: float | None = NotSet,
        zmax: float | None = NotSet,
        zbin_num: int | None = NotSet,
        method: str | None = NotSet,
        zbins: NDArray[np.float_] | None = NotSet,
        # BackendConfig
        thread_num: int | None = NotSet,
        crosspatch: bool | None = NotSet,
        rbin_slop: float | None = NotSet
    ) -> Configuration:
        config = self.to_dict()
        if cosmology is not NotSet:
            if isinstance(cosmology, str):
                cosmology = yaml_to_cosmology(cosmology)
            config["cosmology"] = cosmology_to_yaml(cosmology)
        # ScalesConfig
        if rmin is not NotSet:
            config["scales"]["rmin"] = rmin
        if rmax is not NotSet:
            config["scales"]["rmax"] = rmax
        if rweight is not NotSet:
            config["scales"]["rweight"] = rweight
        if rbin_num is not NotSet:
            config["scales"]["rbin_num"] = rbin_num
        # AutoBinningConfig /  ManualBinningConfig
        if zbins is not NotSet:
            warn_binning_args_ignored(zmin, zmax, zbin_num)
            config["binning"]["zbins"] = zbins
        else:
            if zmin is not NotSet:
                config["binning"]["zmin"] = zmin
            if zmax is not NotSet:
                config["binning"]["zmax"] = zmax
            if zbin_num is not NotSet:
                config["binning"]["zbin_num"] = zbin_num
            if method is not NotSet:
                config["binning"]["method"] = method
        # BackendConfig
        if thread_num is not NotSet:
            config["backend"]["thread_num"] = thread_num
        if crosspatch is not NotSet:
            config["backend"]["crosspatch"] = crosspatch
        if rbin_slop is not NotSet:
            config["backend"]["rbin_slop"] = rbin_slop
        return self.__class__.from_dict(config)

    def plot_scales(
        self,
        catalog: BaseCatalog,
        log: bool = True,
        legend: bool = True
    ) -> Figure:
        import matplotlib.pyplot as plt

        fig, ax_scale = plt.subplots(1, 1)
        # plot scale of annulus
        for r_min, r_max in self.scales.as_array():
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
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        **kwargs
    ) -> Configuration:
        config = {k: v for k, v in the_dict.items()}
        cosmology = parse_cosmology(config.pop(
            "cosmology", DEFAULT.Configuration.cosmology))
        # parse the required subgroups
        try:
            scales = ScalesConfig.from_dict(config.pop("scales"))
        except (TypeError, KeyError) as e:
            parse_section_error(e, "scales")
        try:
            binning_conf = config.pop("binning")
            if "zbins" in binning_conf:
                binning = ManualBinningConfig(binning_conf["zbins"])
            else:
                binning = AutoBinningConfig.generate(
                    cosmology=cosmology, **binning_conf)
        except (TypeError, KeyError) as e:
            parse_section_error(e, "binning")
        # parse the optional subgroups
        try:
            backend = BackendConfig.from_dict(config.pop("backend"))
        except KeyError:
            backend = BackendConfig()
        except TypeError as e:
            parse_section_error(e, "backend")
        # check that there are no entries left
        if len(config) > 0:
            key = next(iter(config.keys()))
            raise ConfigurationError(f"encountered unknown section '{key}'")
        return cls(
            scales=scales, binning=binning,
            backend=backend, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        values = dict()
        for attr in asdict(self):
            value = getattr(self, attr)  # avoid asdict() recursion
            if attr == "cosmology":
                values[attr] = cosmology_to_yaml(value)
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


@dataclass(frozen=True)
class ResamplingConfig(DictRepresentation):

    method: str = DEFAULT.Resampling.method
    crosspatch: bool = DEFAULT.Resampling.crosspatch
    n_boot: int = DEFAULT.Resampling.n_boot
    global_norm: bool = DEFAULT.Resampling.global_norm
    seed: int = DEFAULT.Resampling.seed
    _resampling_idx: NDArray[np.int_] | None = field(
        default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in self.implemented_methods:
            opts = ", ".join(f"'{s}'" for s in self.implemented_methods)
            raise ConfigurationError(
                f"invalid resampling method '{self.method}', "
                f"must either of {opts}")

    @classmethod
    @property
    def implemented_methods(cls) -> tuple[str]:
        return ("jackknife", "bootstrap")

    @property
    def n_patches(self) -> int | None:
        if self._resampling_idx is None:
            return None
        else:
            return self._resampling_idx.shape[1]

    def _generate_bootstrap(self, n_patches: int) -> NDArray[np.int_]:
        N = n_patches
        rng = np.random.default_rng(seed=self.seed)
        return rng.integers(0, N, size=(self.n_boot, N))

    def _generate_jackknife(self, n_patches: int) -> NDArray[np.int_]:
        N = n_patches
        idx = np.delete(np.tile(np.arange(0, N), N), np.s_[::N+1])
        return idx.reshape((N, N-1))

    def get_samples(self, n_patches: int) -> NDArray[np.int_]:
        # generate samples once, afterwards check that n_patches matches
        if self._resampling_idx is None:
            if self.method == "jackknife":
                idx = self._generate_jackknife(n_patches)
            else:
                idx = self._generate_bootstrap(n_patches)
            object.__setattr__(self, "_resampling_idx", idx)
        elif n_patches != self.n_patches:
            raise ValueError(
                f"'n_patches' does not match, expected {self.n_patches}, but "
                f"got {n_patches}")
        return self._resampling_idx

    def reset(self) -> None:
        object.__setattr__(self, "_resampling_idx", None)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> ResamplingConfig:
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        if self.method == "jackknife":
            return dict(method=self.method, crosspatch=self.crosspatch)
        else:
            return super().to_dict()


METHOD_OPTIONS = ResamplingConfig.implemented_methods
"""Names of implemented resampling methods.
"""
