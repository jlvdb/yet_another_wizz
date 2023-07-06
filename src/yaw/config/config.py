from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import yaml

from yaw.core.abc import DictRepresentation
from yaw.core.docs import Parameter
from yaw.core.cosmology import (
    TypeCosmology, get_default_cosmology, r_kpc_to_angle)

from yaw.config import default as DEFAULT, OPTIONS, utils
from yaw.config.backend import BackendConfig
from yaw.config.binning import (
    AutoBinningConfig, ManualBinningConfig, make_binning_config,
    warn_binning_args_ignored)
from yaw.config.scales import ScalesConfig

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from yaw.catalogs import BaseCatalog
    from yaw.core.utils import TypePathStr


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration(DictRepresentation):
    """The central configration for correlation measurements.

    Bundles the configuration of measurement scales, redshift binning, and
    backend parameters in a single, hierarchical configuration class.
    Additionally holds the cosmological model used for distance calculations.

    .. Note::

        The structure and meaning of the parameters is described in more detail
        in the specialised configuration objects :obj:`ScalesConfig`,
        :obj:`AutoBinningConfig` / :obj:`ManualBinningConfig`,
        :obj:`BackendConfig`, which are stored as class attributes
        :obj:`scales`, :obj:`binning`, and :obj:`backend`.

        To access e.g. the lower measurement scale limit, use

        >>> Configuration.scales.rmin
        ...

        which accesses the :obj:`ScalesConfig.rmin` attribute.

    A new instance should be constructed with the :meth:`create` method or
    as a modified variant with the :meth:`modify` method.

    Args:
        scales (:obj:`~yaw.config.ScalesConfig`):
            The configuration of the measurement scales.
        binning (:obj:`~yaw.config.AutoBinningConfig`, :obj:`~yaw.config.ManualBinningConfig`):
            The redshift binning configuration.
        backend (:obj:`~yaw.config.BackendConfig`):
            The backend-specific configuration.
        cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`~yaw.core.cosmology.CustomCosmology`, :obj:`str`, :obj:`None`, optional)
            The cosmological model for distance calculations.
    """

    scales: ScalesConfig
    """The configuration of the measurement scales."""
    binning: AutoBinningConfig | ManualBinningConfig
    """The redshift binning configuration."""
    backend: BackendConfig = field(default_factory=BackendConfig)
    """The backend-specific configuration."""
    cosmology: TypeCosmology | str | None = field(
        default=DEFAULT.Configuration.cosmology,
        metadata=Parameter(
            type=str, choices=OPTIONS.cosmology,
            help="cosmological model used for distance calculations",
            default_text="(see astropy.cosmology, default: %(default)s)"))
    """The cosmological model for distance calculations."""

    def __post_init__(self) -> None:
        # parse cosmology
        if self.cosmology is None:
            cosmology = get_default_cosmology()
        elif isinstance(self.cosmology, str):
            cosmology = utils.yaml_to_cosmology(self.cosmology)
        elif not isinstance(self.cosmology, get_args(TypeCosmology)):
            which = ", ".join(get_args(TypeCosmology))
            raise utils.ConfigError(
                f"'cosmology' must be instance of: {which}")
        else:
            cosmology = self.cosmology
        cosmology = utils.parse_cosmology(self.cosmology)
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
        # AutoBinningConfig / ManualBinningConfig
        zmin: ArrayLike = None,
        zmax: ArrayLike = None,
        zbin_num: int | None = DEFAULT.Configuration.binning.zbin_num,
        method: str = DEFAULT.Configuration.binning.method,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = DEFAULT.Configuration.backend.thread_num,
        crosspatch: bool = DEFAULT.Configuration.backend.crosspatch,
        rbin_slop: float = DEFAULT.Configuration.backend.rbin_slop
    ) -> Configuration:
        """Create a new configuration object.

        Except for the ``cosmology`` parameter, all other parameters are passed
        to the constructors of the respective :obj:`ScalesConfig`,
        :obj:`AutoBinningConfig` / :obj:`ManualBinningConfig`,
        :obj:`BackendConfig` classes.

        .. Note::

            If custom bin edges are provided through the ``zbins`` parameter,
            ``zmin``, ``zmax``, ``zbin_num`` (optional), and ``method``
            (optional) are ignored. Otherwise, at least ``zmin``, ``zmax`` are
            required and a binning will be generated automatically.

        Otherwise, only ``rmin`` and ``rmax`` are required arguments, e.g.:

        >>> yaw.Configuration.create(rmin=100, rmax=1000, zmin=0.1, zmax=1.0)

        Keyword Args:
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`~yaw.core.cosmology.CustomCosmology`, :obj:`str`, :obj:`None`, optional):
                Named astropy cosmology used to compute distances. For options
                see :obj:`~yaw.config.options.Options.cosmology`.
            rmin (:obj:`ArrayLike`):
                (List of) lower scale limit in kpc (pyhsical).
            rmax (:obj:`ArrayLike`):
                (List of) upper scale limit in kpc (pyhsical).
            rweight (:obj:`float`, optional):
                Weight galaxy pairs by their separation to power 'rweight'.
            rbin_num (:obj:`int`, optional):
                Number of bins in log r used (i.e. resolution) to compute
                distance weights.
            zmin (:obj:`float`):
                Lower redshift limit.
            zmax (:obj:`float`):
                Upper redshift limit.
            zbin_num (:obj:`int`, optional):
                Number of redshift bins
            method (:obj:`str`, optional):
                Method used to generate the redshift binning. For options see
                :obj:`~yaw.config.options.Options.binning`.
            zbins (:obj:`NDArray`, optional):
                Manually define redshift bin edges.
            thread_num (:obj:`int`, optional):
                Default number of threads to use.
            crosspatch (:obj:`bool`, optional):
                whether to count pairs across patch boundaries (scipy backend
                only)
            rbin_slop (:obj:`float`, optional):
                TreeCorr 'rbin_slop' parameter
    
        Returns:
            :obj:`Configuration`
        """
        cosmology = utils.parse_cosmology(cosmology)
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
        cosmology: TypeCosmology | str | None = DEFAULT.NotSet,
        # ScalesConfig
        rmin: ArrayLike | None = DEFAULT.NotSet,
        rmax: ArrayLike | None = DEFAULT.NotSet,
        rweight: float | None = DEFAULT.NotSet,
        rbin_num: int | None = DEFAULT.NotSet,
        # AutoBinningConfig /  ManualBinningConfig
        zmin: float | None = DEFAULT.NotSet,
        zmax: float | None = DEFAULT.NotSet,
        zbin_num: int | None = DEFAULT.NotSet,
        method: str | None = DEFAULT.NotSet,
        zbins: NDArray[np.float_] | None = DEFAULT.NotSet,
        # BackendConfig
        thread_num: int | None = DEFAULT.NotSet,
        crosspatch: bool | None = DEFAULT.NotSet,
        rbin_slop: float | None = DEFAULT.NotSet
    ) -> Configuration:
        """Create a copy of the current configuration with updated parameter
        values.

        The method arguments are identical to :meth:`create`. Values that should
        not be modified are by default represented by the special value
        :obj:`~yaw.config.default.NotSet`.
        """
        config = self.to_dict()
        if cosmology is not DEFAULT.NotSet:
            if isinstance(cosmology, str):
                cosmology = utils.yaml_to_cosmology(cosmology)
            config["cosmology"] = utils.cosmology_to_yaml(cosmology)
        # ScalesConfig
        if rmin is not DEFAULT.NotSet:
            config["scales"]["rmin"] = rmin
        if rmax is not DEFAULT.NotSet:
            config["scales"]["rmax"] = rmax
        if rweight is not DEFAULT.NotSet:
            config["scales"]["rweight"] = rweight
        if rbin_num is not DEFAULT.NotSet:
            config["scales"]["rbin_num"] = rbin_num
        # AutoBinningConfig /  ManualBinningConfig
        if zbins is not DEFAULT.NotSet:
            warn_binning_args_ignored(zmin, zmax, zbin_num)
            config["binning"]["zbins"] = zbins
        else:
            if zmin is not DEFAULT.NotSet:
                config["binning"]["zmin"] = zmin
            if zmax is not DEFAULT.NotSet:
                config["binning"]["zmax"] = zmax
            if zbin_num is not DEFAULT.NotSet:
                config["binning"]["zbin_num"] = zbin_num
            if method is not DEFAULT.NotSet:
                config["binning"]["method"] = method
        # BackendConfig
        if thread_num is not DEFAULT.NotSet:
            config["backend"]["thread_num"] = thread_num
        if crosspatch is not DEFAULT.NotSet:
            config["backend"]["crosspatch"] = crosspatch
        if rbin_slop is not DEFAULT.NotSet:
            config["backend"]["rbin_slop"] = rbin_slop
        return self.__class__.from_dict(config)

    def plot_scales(
        self,
        catalog: BaseCatalog,
        log: bool = True,
        legend: bool = True
    ) -> Figure:
        """Plot the configured correlation scales at different redshifts in
        comparison to the size of patches in a data catalogue."""
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
        cosmology = utils.parse_cosmology(config.pop(
            "cosmology", DEFAULT.Configuration.cosmology))
        # parse the required subgroups
        try:
            scales = ScalesConfig.from_dict(config.pop("scales"))
        except (TypeError, KeyError) as e:
            utils.parse_section_error(e, "scales")
        try:
            binning_conf = config.pop("binning")
            if "zbins" in binning_conf:
                binning = ManualBinningConfig(binning_conf["zbins"])
            else:
                binning = AutoBinningConfig.generate(
                    cosmology=cosmology, **binning_conf)
        except (TypeError, KeyError) as e:
            utils.parse_section_error(e, "binning")
        # parse the optional subgroups
        try:
            backend = BackendConfig.from_dict(config.pop("backend"))
        except KeyError:
            backend = BackendConfig()
        except TypeError as e:
            utils.parse_section_error(e, "backend")
        # check that there are no entries left
        if len(config) > 0:
            key = next(iter(config.keys()))
            raise utils.ConfigError(f"encountered unknown section '{key}'")
        return cls(
            scales=scales, binning=binning,
            backend=backend, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        values = dict()
        for attr in asdict(self):
            value = getattr(self, attr)  # avoid asdict() recursion
            if attr == "cosmology":
                values[attr] = utils.cosmology_to_yaml(value)
            else:
                values[attr] = value.to_dict()
        return values

    @classmethod
    def from_yaml(cls, path: TypePathStr) -> Configuration:
        """Create a new instance by loading the configuration from a YAML file.
        
        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
                Path to the YAML file containing the configuration.

        Returns:
            :obj:`Configuration`                
        """
        logger.info(f"reading configuration file '{path}'")
        with open(str(path)) as f:
            config = yaml.safe_load(f.read())
        return cls.from_dict(config)

    def to_yaml(self, path: TypePathStr) -> None:
        """Store the configuration as YAML file.
        
        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
                Path to which the YAML file is written.
        """
        logger.info(f"writing configuration file '{path}'")
        string = yaml.dump(self.to_dict())
        with open(str(path), "w") as f:
            f.write(string)
