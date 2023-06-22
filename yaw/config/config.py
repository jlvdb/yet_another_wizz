from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, get_args

import astropy.cosmology
import numpy as np
import yaml

from yaw.core import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.default import NotSet
from yaw.core.docs import Parameter
from yaw.core.cosmology import (
    COSMOLOGY_OPTIONS, TypeCosmology, get_default_cosmology, r_kpc_to_angle)

from yaw.config.backend import BackendConfig
from yaw.config.binning import (
    AutoBinningConfig, ManualBinningConfig, make_binning_config,
    warn_binning_args_ignored)
from yaw.config.scales import ScalesConfig
from yaw.config.utils import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from yaw.catalogs import BaseCatalog


logger = logging.getLogger(__name__)


def cosmology_to_yaml(cosmology: TypeCosmology) -> str:
    if not isinstance(cosmology, astropy.cosmology.FLRW):
        raise ConfigError("cannot serialise custom cosmoligies to YAML")
    if cosmology.name not in astropy.cosmology.available:
        raise ConfigError(
            "can only serialise predefined astropy cosmologies to YAML")
    return cosmology.name


def yaml_to_cosmology(cosmo_name: str) -> TypeCosmology:
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigError(
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
        raise ConfigError(
            f"'cosmology' must be instance of: {which}")
    return cosmology


def parse_section_error(
    exception: Exception,
    section: str,
    reraise: Exception = ConfigError
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


@dataclass(frozen=True)
class Config(DictRepresentation):
    """The central configration for correlation measurements.

    Construct with .create() method.
    """

    scales: ScalesConfig
    binning: AutoBinningConfig | ManualBinningConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    cosmology: TypeCosmology | str | None = field(
        default=DEFAULT.Config.cosmology,
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
            raise ConfigError(
                f"'cosmology' must be instance of: {which}")
        else:
            cosmology = self.cosmology
        cosmology = parse_cosmology(self.cosmology)
        object.__setattr__(self, "cosmology", cosmology)

    @classmethod
    def create(
        cls,
        *,
        cosmology: TypeCosmology | str | None = DEFAULT.Config.cosmology,
        # ScalesConfig
        rmin: ArrayLike,
        rmax: ArrayLike,
        rweight: float | None = DEFAULT.Config.scales.rweight,
        rbin_num: int = DEFAULT.Config.scales.rbin_num,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: ArrayLike = None,
        zmax: ArrayLike = None,
        zbin_num: int | None = DEFAULT.Config.binning.zbin_num,
        method: str = DEFAULT.Config.binning.method,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = DEFAULT.Config.backend.thread_num,
        crosspatch: bool = DEFAULT.Config.backend.crosspatch,
        rbin_slop: float = DEFAULT.Config.backend.rbin_slop
    ) -> Config:
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
            :obj:`Config`
        
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
    ) -> Config:
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
    ) -> Config:
        config = {k: v for k, v in the_dict.items()}
        cosmology = parse_cosmology(config.pop(
            "cosmology", DEFAULT.Config.cosmology))
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
            raise ConfigError(f"encountered unknown section '{key}'")
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
    def from_yaml(cls, path: str) -> Config:
        logger.info(f"reading configuration file '{path}'")
        with open(path) as f:
            config = yaml.safe_load(f.read())
        return cls.from_dict(config)

    def to_yaml(self, path: str) -> None:
        logger.info(f"writing configuration file '{path}'")
        string = yaml.dump(self.to_dict())
        with open(path, "w") as f:
            f.write(string)
