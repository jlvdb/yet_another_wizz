from __future__ import annotations

import logging
import pprint
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, get_args

import astropy.cosmology
import numpy as np
from astropy import units

from yaw import parallel
from yaw.abc import YamlSerialisable
from yaw.containers import Binning
from yaw.cosmology import (
    CustomCosmology,
    TypeCosmology,
    cosmology_is_equal,
    get_default_cosmology,
)
from yaw.options import BinMethod, BinMethodAuto, Closed, get_options

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import Any, TypeVar

    from numpy.typing import NDArray

    T = TypeVar("T")
    TypeBaseConfig = TypeVar("TypeBaseConfig", bound="BaseConfig")

__all__ = [
    "BinningConfig",
    "Configuration",
    "ScalesConfig",
]

logger = logging.getLogger(__name__)


class _NotSet_meta(type):
    def __repr__(self) -> str:
        return "NotSet"  # pragma: no cover

    def __bool__(self) -> bool:
        return False


class NotSet(metaclass=_NotSet_meta):
    """Placeholder for configuration values that are not set."""

    pass


class ConfigError(Exception):
    pass


def cosmology_to_yaml(cosmology: TypeCosmology) -> str:
    """
    Attempt to serialise a cosmology instance to YAML.

    .. caution::
        This currently works only for named astropy cosmologies.

    Args:
        cosmology:
            A cosmology class instance, either a custom or astropy cosmology.

    Returns:
        A YAML string.
    """
    if isinstance(cosmology, CustomCosmology):
        raise ConfigError("cannot serialise custom cosmologies to YAML")

    elif not isinstance(cosmology, astropy.cosmology.FLRW):
        raise TypeError(f"invalid type '{type(cosmology)}' for cosmology")

    if cosmology.name not in astropy.cosmology.available:
        raise ConfigError("can only serialise predefined astropy cosmologies to YAML")

    return cosmology.name


def yaml_to_cosmology(cosmo_name: str) -> TypeCosmology:
    """Restore a cosmology class instance from a YAML string."""
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigError(
            "unknown cosmology, for available options see 'astropy.cosmology.available'"
        )

    return getattr(astropy.cosmology, cosmo_name)


def parse_cosmology(cosmology: TypeCosmology | str | None) -> TypeCosmology:
    """
    Parse and verify that the provided cosmology is supported by
    yet_another_wizz.

    Parses any YAML string or replaces None with the default cosmology.

    Args:
        cosmology:
            A cosmology class instance, either a custom or astropy cosmology, or
            ``None`` or a cosmology serialised to YAML.

    Returns:
        A cosmology class instance, either a custom or astropy cosmology.

    Raises:
        ConfigError:
            If the cosmology cannot be parsed.
    """
    if cosmology is None:
        return get_default_cosmology()

    elif isinstance(cosmology, str):
        return yaml_to_cosmology(cosmology)

    elif not isinstance(cosmology, get_args(TypeCosmology)):
        which = ", ".join(str(c) for c in get_args(TypeCosmology))
        raise ConfigError(f"'cosmology' must be instance of: {which}")

    return cosmology


class Immutable:
    """Meta-class for configuration classes that prevent mutating attributes."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"attribute '{name}' is immutable")


@dataclass
class Parameter:
    """Defines the meta data for a configuration parameter, including a
    describing help message."""

    name: str
    help: str
    type: type
    is_sequence: bool = field(default=False)
    default: Any = field(default=NotSet)
    choices: tuple[Any] = field(default=NotSet)

    def to_dict(self) -> dict[str, Any]:  # NOTE: used by RAIL wrapper
        return {key: val for key, val in asdict(self).items() if val is not NotSet}


class ParamSpec(Mapping[str, Parameter]):
    """Dict-like collection of configuration parameters."""

    def __init__(self, params: Iterable[Parameter]) -> None:
        self._params = {p.name: p for p in params}

    def __str__(self) -> str:
        string = f"{type(self).__name__}:"
        for value in self.values():
            string += f"\n  {value}"
        return string

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __iter__(self) -> Iterator[str]:
        yield from iter(self._params)

    def __contains__(self, item) -> bool:
        return item in self._params


class BaseConfig(YamlSerialisable):
    """
    Meta-class for all configuration classes.

    Implements basic interface that allows serialisation to YAML and methods to
    create or modify and existing configuration class instance without mutating
    the original.
    """

    @classmethod
    def from_dict(
        cls: type[TypeBaseConfig],
        the_dict: dict[str, Any],
    ) -> TypeBaseConfig:
        return cls.create(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def create(cls: type[TypeBaseConfig], **kwargs: Any) -> TypeBaseConfig:
        """Create a new instance with the given parameter values."""
        pass

    @abstractmethod
    def modify(self: TypeBaseConfig, **kwargs: Any | NotSet) -> TypeBaseConfig:
        """Create a new instance by modifing the original instance with the
        given parameter values."""
        conf_dict = self.to_dict()
        conf_dict.update(
            {key: value for key, value in kwargs.items() if value is not NotSet}
        )
        return type(self).from_dict(conf_dict)

    def __repr__(self) -> str:
        return pprint.pformat(self.to_dict())

    @abstractmethod
    def __eq__(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def get_paramspec(cls) -> ParamSpec:
        """
        Generate a listing of parameters that may be used by external tool
        to auto-generate an interface to this configuration class.

        Returns:
            A :obj:`ParamSpec` instance, that is a key-value mapping from
            parameter name to the parameter meta data for this configuration
            class.
        """
        pass


def parse_optional(value: Any | None, type: type[T]) -> T | None:
    """Instantiate a type with a given value if the value is not ``None``."""
    if value is None:
        return None

    return type(value)


class ScalesConfig(BaseConfig, Immutable):
    """
    Configuration for correlation function measurement scales.

    Correlations are measured by counting all pairs between a minimum and
    maximum physical scale given in kpc. The code can be configured with either
    a single scale range or multiple (overlapping) scale ranges.

    Additionally, pair counts can be weighted by the separation distance using a
    power law :math:`w(r) \\propto r^\\alpha`. For performance reasons, the pair
    counts are not weighted indivudially but in fine logarithmic bins of angular
    separation, i.e. :math:`w(r) \\sim w(r_i)`, where :math:`r_i` is the
    logarithmic center of the :math:`i`-th bin.

    .. note::
        The preferred way to create a new configuration instance is using the
        :meth:`create()` constructor.

        All configuration objects are immutable. To modify an existing
        configuration, create a new instance with updated values by using the
        :meth:`modify()` method.
    """

    rmin: list[float] | float
    """Single or multiple lower scale limits in kpc (angular diameter
    distance)."""
    rmax: list[float] | float
    """Single or multiple upper scale limits in kpc (angular diameter
    distance)."""
    rweight: float | None
    """Optional power-law exponent :math:`\\alpha` used to weight pairs by their
    separation."""
    resolution: int | None
    """Optional number of radial logarithmic bin used to approximate the
    weighting by separation."""

    def __init__(
        self,
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        *,
        rweight: float | None = None,
        resolution: int | None = None,
    ) -> None:
        rmin: NDArray = np.atleast_1d(rmin)
        rmax: NDArray = np.atleast_1d(rmax)

        if rmin.ndim != rmax.ndim and rmin.ndim != 1:
            raise ConfigError("'rmin/rmax' must be scalars or one-dimensional arrays")
        if len(rmin) != len(rmax):
            raise ConfigError("number of elements in 'rmin' and 'rmax' does not match")

        if np.any((rmax - rmin) <= 0.0):
            raise ConfigError("'rmin' must be smaller than corresponding 'rmax'")

        # ensure a YAML-friendly, i.e. native python, type
        object.__setattr__(self, "rmin", rmin.squeeze().tolist())
        object.__setattr__(self, "rmax", rmax.squeeze().tolist())
        object.__setattr__(self, "rweight", parse_optional(rweight, float))
        object.__setattr__(self, "resolution", parse_optional(resolution, int))

    @property
    def num_scales(self) -> int:
        """Number of correlation scales."""
        try:
            return len(self.rmin)
        except TypeError:
            return 1

    def to_dict(self) -> dict[str, Any]:
        attrs = ("rmin", "rmax", "rweight", "resolution")
        return {attr: getattr(self, attr) for attr in attrs}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            np.array_equal(self.rmin, other.rmin)
            and np.array_equal(self.rmax, other.rmax)
            and self.rweight == other.rweight
            and self.rbin_num == other.rbin_num
        )

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = [
            Parameter(
                name="rmin",
                help="Single or multiple lower scale limits in kpc (angular diameter distance).",
                type=float,
                is_sequence=True,
            ),
            Parameter(
                name="rmax",
                help="Single or multiple upper scale limits in kpc (angular diameter distance).",
                type=float,
                is_sequence=True,
            ),
            Parameter(
                name="rweight",
                help="Optional power-law exponent :math:`\\alpha` used to weight pairs by their separation.",
                type=float,
                default=None,
            ),
            Parameter(
                name="resolution",
                help="Optional number of radial logarithmic bin used to approximate the weighting by separation.",
                type=int,
                default=None,
            ),
        ]
        return ParamSpec(params)

    @classmethod
    def create(
        cls,
        *,
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        rweight: float | None = None,
        resolution: int | None = None,
    ) -> ScalesConfig:
        """
        Create a new instance with the given parameters.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in kpc (angular diameter
                distance).
            rmax:
                Single or multiple upper scale limits in kpc (angular diameter
                distance).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.

        Returns:
            New configuration instance.
        """
        return cls(rmin=rmin, rmax=rmax, rweight=rweight, resolution=resolution)

    def modify(
        self,
        *,
        rmin: Iterable[float] | float = NotSet,
        rmax: Iterable[float] | float = NotSet,
        rweight: float | None = NotSet,
        resolution: int | None = NotSet,
    ) -> ScalesConfig:
        """
        Create a new configuration instance with updated parameter values.

        Parameter values are only updated if they are provided as inputs to this
        function, otherwise they are retained from the original instance.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in kpc (angular diameter
                distance).
            rmax:
                Single or multiple upper scale limits in kpc (angular diameter
                distance).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.

        Returns:
            New instance with updated parameter values.
        """
        return super().modify(rmin, rmax, rweight, resolution)


class RedshiftBinningFactory:
    """Simple factory class to create redshift binnings. Takes an optional
    cosmology as input for distance conversions."""

    def __init__(self, cosmology: TypeCosmology | None = None) -> None:
        self.cosmology = cosmology or get_default_cosmology()

    def linear(
        self,
        min: float,
        max: float,
        num_bins: int,
        *,
        closed: Closed | str = Closed.right,
    ) -> Binning:
        """Creates a linear redshift binning between a min and max redshift."""
        edges = np.linspace(min, max, num_bins + 1)
        return Binning(edges, closed=closed)

    def comoving(
        self,
        min: float,
        max: float,
        num_bins: int,
        *,
        closed: Closed | str = Closed.right,
    ) -> Binning:
        """Creates a binning linear in comoving distance between a min and max
        redshift."""
        comov_min, comov_cmax = self.cosmology.comoving_distance([min, max])
        comov_edges = np.linspace(comov_min, comov_cmax, num_bins + 1)
        if not isinstance(comov_edges, units.Quantity):
            comov_edges = comov_edges * units.Mpc

        edges = astropy.cosmology.z_at_value(
            self.cosmology.comoving_distance, comov_edges
        )
        return Binning(edges.value, closed=closed)

    def logspace(
        self,
        min: float,
        max: float,
        num_bins: int,
        *,
        closed: Closed | str = Closed.right,
    ) -> Binning:
        """Creates a binning linear in 1+ln(z) between a min and max redshift."""
        log_min, log_max = np.log([1.0 + min, 1.0 + max])
        edges = np.logspace(log_min, log_max, num_bins + 1, base=np.e) - 1.0
        return Binning(edges, closed=closed)

    def get_method(
        self, method: BinMethodAuto | str = BinMethodAuto.linear
    ) -> Callable[..., Binning]:
        """Use a string identifier to select the desired factory method."""
        return getattr(self, BinMethodAuto(method))


class BinningConfig(BaseConfig, Immutable):
    """
    Configuration of the redshift binning for correlation function measurements.

    Correlations are measured in bins of redshift, which determines the
    redshift-resolution of the clustering redshift estimate. This configuration
    class offers three automatic methods to generate these bins between a
    minimum and maximum redshift:

    - ``linear`` (default): bin edges spaced linearly in redshift :math:`z`,
    - ``comoving``: bin edges spaced linearly in comoving distance
      :math:`\\chi(z)`, and
    - ``logspace``: bin edges spaced linearly in :math:`1+\\ln(z)`.

    Alternatively, custom bin edges may be provided.

    .. note::
        The preferred way to create a new configuration instance is using the
        :meth:`create()` constructor.

        All configuration objects are immutable. To modify an existing
        configuration, create a new instance with updated values by using the
        :meth:`modify()` method. The bin edges are recomputed when necessary.
    """

    binning: Binning
    """Container for the redshift bins."""
    method: BinMethod
    """Method used to generate the bin edges, must be either of ``linear``,
    ``comoving``, ``logspace``, or ``custom``."""

    def __init__(
        self,
        binning: Binning,
        method: BinMethod | str = BinMethod.linear,
    ) -> None:
        if not isinstance(binning, Binning):
            raise TypeError(f"'binning' must be of type '{type(binning)}'")
        object.__setattr__(self, "binning", binning)
        object.__setattr__(self, "method", BinMethod(method))

    @property
    def edges(self) -> NDArray:
        """Array of redshift bin edges."""
        return self.binning.edges

    @property
    def zmin(self) -> float:
        """Lowest redshift bin edge."""
        return float(self.binning.edges[0])

    @property
    def zmax(self) -> float:
        """Highest redshift bin edge."""
        return float(self.binning.edges[-1])

    @property
    def num_bins(self) -> int:
        """Number of redshift bins."""
        return len(self.binning)

    @property
    def closed(self) -> Closed:
        """String indicating if the bin edges are closed on the ``left`` or the
        ``right`` side."""
        return self.binning.closed

    @property
    def is_custom(self) -> bool:
        """Whether the bin edges are provided by the user."""
        return self.method == "custom"

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, Any], cosmology: TypeCosmology | None = None
    ) -> BinningConfig:
        """
        Restore the class instance from a python dictionary.

        Args:
            the_dict:
                Dictionary containing all required data attributes to restore
                the instance, see also :meth:`to_dict()`.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            Restored class instance.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        if the_dict["method"] == "custom":
            edges = the_dict.pop("edges")
            closed = the_dict.pop("closed")
            binning = Binning(edges, closed=closed)
            return cls(binning, **the_dict)

        return cls.create(**the_dict, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        if self.is_custom:
            the_dict = dict(edges=self.binning.edges)

        else:
            the_dict = dict(
                zmin=self.zmin,
                zmax=self.zmax,
                num_bins=self.num_bins,
            )

        the_dict["method"] = str(self.method)
        the_dict["closed"] = str(self.closed)
        return the_dict

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.method == other.method and self.binning == other.binning

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = [
            Parameter(
                name="zmin",
                help="Lowest redshift bin edge to generate.",
                type=float,
            ),
            Parameter(
                name="zmax",
                help="Highest redshift bin edge to generate.",
                type=float,
            ),
            Parameter(
                name="num_bins",
                help="Number of redshift bins to generate.",
                type=int,
                default=30,
            ),
            Parameter(
                name="method",
                help="Method used to generate the bin edges, must be either of ``linear``, ``comoving``, ``logspace``, or ``custom``.",
                type=str,
                choices=get_options(Closed),
                default=BinMethod.linear,
            ),
            Parameter(
                name="edges",
                help="Use these custom bin edges instead of generating them.",
                type=float,
                is_sequence=True,
                default=None,
            ),
            Parameter(
                name="closed",
                help="String indicating if the bin edges are closed on the ``left`` or the ``right`` side.",
                type=str,
                choices=get_options(Closed),
                default=str(Closed.right),
            ),
        ]
        return ParamSpec(params)

    @classmethod
    def create(
        cls,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
        num_bins: int = 30,
        method: BinMethod | str = BinMethod.linear,
        edges: Iterable[float] | None = None,
        closed: Closed | str = Closed.right,
        cosmology: TypeCosmology | str | None = None,
    ) -> BinningConfig:
        """
        Create a new instance with the given parameters.

        Keyword Args:
            zmin:
                Lowest redshift bin edge to generate.
            zmax:
                Highest redshift bin edge to generate.
            num_bins:
                Number of redshift bins to generate.
            method:
                Method used to generate the bin edges, must be either of
                ``linear``, ``comoving``, ``logspace``, or ``custom``.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                String indicating if the bin edges are closed on the ``left`` or
                the ``right`` side.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            New configuration instance.

        .. note::
            All function parameters are optional, but either ``zmin`` and
            ``zmax`` (generate bin edges), or ``edges`` (custom bin edges) must
            be provided.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        auto_args_set = (zmin is not None, zmax is not None)
        custom_args_set = (edges is not None,)
        if not all(custom_args_set) and not all(auto_args_set):
            raise ConfigError("either 'edges' or 'zmin' and 'zmax' are required")

        closed = Closed(closed)

        if all(auto_args_set):  # generate bin edges
            if all(custom_args_set):
                warnings.warn(
                    "'zbins' set but ignored since 'zmin' and 'zmax' are provided"
                )
            method = BinMethod(method)
            bin_func = RedshiftBinningFactory(cosmology).get_method(method)
            binning = bin_func(zmin, zmax, num_bins, closed=closed)

        else:  # use provided bin edges
            method = BinMethod.custom
            binning = Binning(edges, closed=closed)

        return cls(binning, method=method)

    def modify(
        self,
        *,
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = NotSet,
        method: BinMethod | str | NotSet = NotSet,
        edges: Iterable[float] | NotSet = NotSet,
        closed: Closed | str | NotSet = NotSet,
        cosmology: TypeCosmology | str | None | NotSet = NotSet,
    ) -> BinningConfig:
        """
        Create a new configuration instance with updated parameter values.

        Parameter values are only updated if they are provided as inputs to this
        function, otherwise they are retained from the original instance.

        Keyword Args:
            zmin:
                Lowest redshift bin edge to generate.
            zmax:
                Highest redshift bin edge to generate.
            num_bins:
                Number of redshift bins to generate.
            method:
                Method used to generate the bin edges, must be either of
                ``linear``, ``comoving``, ``logspace``, or ``custom``.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                String indicating if the bin edges are closed on the ``left`` or
                the ``right`` side.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            New instance with updated redshift bins.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        if edges is NotSet:
            if method == "custom":
                raise ConfigError("'method' is 'custom' but no bin edges provided")
            the_dict = dict()
            the_dict["zmin"] = self.zmin if zmin is NotSet else zmin
            the_dict["zmax"] = self.zmax if zmax is NotSet else zmax
            the_dict["num_bins"] = self.num_bins if num_bins is NotSet else num_bins
            the_dict["method"] = self.method if method is NotSet else BinMethod(method)

        else:
            the_dict = dict(edges=edges)
            the_dict["method"] = BinMethod.custom

        the_dict["method"] = str(the_dict["method"])
        the_dict["closed"] = str(self.closed if closed is NotSet else Closed(closed))

        return type(self).from_dict(the_dict, cosmology=cosmology)


default_cosmology = get_default_cosmology().name


class Configuration(BaseConfig, Immutable):
    """
    Configuration for correlation function measurements.

    This is the top-level configuration class for `yet_another_wizz`, defining
    correlation scales, redshift binning, and the cosmological model used for
    distance calculations.

    .. note::
        The preferred way to create a new configuration instance is using the
        :meth:`create()` constructor.

        All configuration objects are immutable. To modify an existing
        configuration, create a new instance with updated values by using the
        :meth:`modify()` method. The bin edges are recomputed when necessary.
    """

    scales: ScalesConfig
    """Organises the configuration of correlation scales."""
    binning: BinningConfig
    """Organises the configuration of redshift bins."""
    cosmology: TypeCosmology | str
    """Optional cosmological model to use for distance computations."""
    max_workers: int | None
    """Limit the number of workers for parallel operations (all by default)."""

    def __init__(
        self,
        scales: ScalesConfig,
        binning: BinningConfig,
        cosmology: TypeCosmology | str | None = None,
        max_workers: int | None = None,
    ) -> None:
        if not isinstance(scales, ScalesConfig):
            raise TypeError(f"'scales' must be of type '{type(ScalesConfig)}'")
        object.__setattr__(self, "scales", scales)

        if not isinstance(binning, BinningConfig):
            raise TypeError(f"'binning' must be of type '{type(BinningConfig)}'")
        object.__setattr__(self, "binning", binning)

        object.__setattr__(self, "cosmology", parse_cosmology(cosmology))
        object.__setattr__(
            self, "max_workers", int(max_workers) if max_workers else None
        )

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> Configuration:
        the_dict = the_dict.copy()

        cosmology = parse_cosmology(the_dict.pop("cosmology", default_cosmology))
        max_workers = the_dict.pop("max_workers", None)

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

        return cls(
            scales=scales, binning=binning, cosmology=cosmology, max_workers=max_workers
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            scales=self.scales.to_dict(),
            binning=self.binning.to_dict(),
            cosmology=cosmology_to_yaml(self.cosmology),
            max_workers=self.max_workers,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> Configuration:
        new = None

        if parallel.on_root():
            logger.info("reading configuration file: %s", path)

            new = super().from_file(path)

        return parallel.COMM.bcast(new, root=0)

    def to_file(self, path: Path | str) -> None:
        if parallel.on_root():
            logger.info("writing configuration file: %s", path)

            super().to_file(path)

        parallel.COMM.Barrier()

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            self.binning == other.binning
            and self.scales == other.scales
            and cosmology_is_equal(self.cosmology, other.cosmology)
        )

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = [
            Parameter(
                name="cosmology",
                help="Optional cosmological model to use for distance computations.",
                type=str,
                default=default_cosmology,
            ),
        ]
        return ParamSpec(params)

    @classmethod
    def create(
        cls,
        *,
        # ScalesConfig
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        rweight: float | None = None,
        resolution: int | None = None,
        # BinningConfig
        zmin: float | None = None,
        zmax: float | None = None,
        num_bins: int = 30,
        method: BinMethod | str = BinMethod.linear,
        edges: Iterable[float] | None = None,
        closed: Closed | str = Closed.right,
        # uncategorized
        cosmology: TypeCosmology | str | None = default_cosmology,
        max_workers: int | None = None,
    ) -> Configuration:
        """
        Create a new instance with the given parameters.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in kpc (angular diameter
                distance).
            rmax:
                Single or multiple upper scale limits in kpc (angular diameter
                distance).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.
            zmin:
                Lowest redshift bin edge to generate.
            zmax:
                Highest redshift bin edge to generate.
            num_bins:
                Number of redshift bins to generate.
            method:
                Method used to generate the bin edges, must be either of
                ``linear``, ``comoving``, ``logspace``, or ``custom``.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                String indicating if the bin edges are closed on the ``left`` or
                the ``right`` side.
            cosmology:
                Optional cosmological model to use for distance computations.
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).

        Returns:
            New configuration instance.

        .. note::
            Although the function parameters are optional, either ``zmin`` and
            ``zmax`` (generate bin edges), or ``edges`` (custom bin edges) must
            be provided.
        """
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

        cosmology = parse_cosmology(cosmology)

        return cls(
            scales=scales, binning=binning, cosmology=cosmology, max_workers=max_workers
        )

    def modify(
        self,
        *,
        # ScalesConfig
        rmin: Iterable[float] | float | NotSet = NotSet,
        rmax: Iterable[float] | float | NotSet = NotSet,
        rweight: float | None | NotSet = NotSet,
        resolution: int | None | NotSet = NotSet,
        # BinningConfig
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = NotSet,
        method: BinMethod | str | NotSet = NotSet,
        edges: Iterable[float] | None = NotSet,
        closed: Closed | str | NotSet = NotSet,
        # uncategorized
        cosmology: TypeCosmology | str | None | NotSet = NotSet,
        max_workers: int | None | NotSet = NotSet,
    ) -> Configuration:
        """
        Create a new configuration instance with updated parameter values.

        Parameter values are only updated if they are provided as inputs to this
        function and retained from the original instance otherwise.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in kpc (angular diameter
                distance).
            rmax:
                Single or multiple upper scale limits in kpc (angular diameter
                distance).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.
            zmin:
                Lowest redshift bin edge to generate.
            zmax:
                Highest redshift bin edge to generate.
            num_bins:
                Number of redshift bins to generate.
            method:
                Method used to generate the bin edges, must be either of
                ``linear``, ``comoving``, ``logspace``, or ``custom``.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                String indicating if the bin edges are closed on the ``left`` or
                the ``right`` side.
            cosmology:
                Optional cosmological model to use for distance computations.
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).

        Returns:
            New instance with updated parameter values.
        """
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

        cosmology = (
            self.cosmology if cosmology is NotSet else parse_cosmology(cosmology)
        )
        max_workers = self.max_workers if max_workers is NotSet else max_workers

        return type(self)(
            scales=scales, binning=binning, cosmology=cosmology, max_workers=max_workers
        )
