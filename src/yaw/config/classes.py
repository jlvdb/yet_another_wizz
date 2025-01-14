"""
Implements the central configuration classes for `yet_another_wizz`. These store
the configuration of measurement scales, redshift binning and cosmological model
to use for correlation fuction measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

import astropy.cosmology
import numpy as np

from yaw.binning import Binning
from yaw.config.base import (
    ConfigError,
    ConfigSection,
    Parameter,
    SequenceParameter,
    YawConfig,
    raise_configerror_with_level,
)
from yaw.cosmology import (
    CustomCosmology,
    RedshiftBinningFactory,
    Scales,
    TypeCosmology,
    cosmology_is_equal,
    get_default_cosmology,
    new_scales,
)
from yaw.options import BinMethod, Closed, NotSet, Unit
from yaw.utils import parallel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

__all__ = [
    "BinningConfig",
    "Configuration",
    "ScalesConfig",
]

logger = logging.getLogger(__name__.removesuffix(".classes"))


@dataclass(frozen=True)
class ScalesConfig(YawConfig):
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

    _paramspec = (
        SequenceParameter(
            name="rmin",
            help="Single or sequence of lower scale limits in given 'unit'.",
            type=float,
        ),
        SequenceParameter(
            name="rmax",
            help="Single or sequence of upper scale limits in given 'unit'.",
            type=float,
        ),
        Parameter(
            name="unit",
            help="The unit of the lower and upper scale limits.",
            type=str,
            choices=Unit,
            default=Unit.kpc,
        ),
        Parameter(
            name="rweight",
            help="Power-law exponent used to weight pairs by their separation.",
            type=float,
            default=None,
        ),
        Parameter(
            name="resolution",
            help="Number of radial logarithmic bin used to approximate the weighting by separation.",
            type=int,
            default=None,
        ),
    )

    scales: Scales
    """Correlation scales in angular or physical units."""
    rweight: float | None
    """Power-law exponent used to weight pairs by their separation."""
    resolution: int | None
    """Number of radial logarithmic bin used to approximate the weighting by
    separation."""

    @property
    def rmin(self) -> float | list[float]:
        """Single or sequence of lower scale limits in given 'unit'."""
        return self.scales.scale_min.squeeze().tolist()

    @property
    def rmax(self) -> float | list[float]:
        """Single or sequence of upper scale limits in given 'unit'."""
        return self.scales.scale_max.squeeze().tolist()

    @property
    def unit(self) -> str:
        """The unit of the lower and upper scale limits (default: kpc)."""
        return str(self.scales.unit)

    @property
    def num_scales(self) -> int:
        """Number of correlation scales."""
        return self.scales.num_scales

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            np.array_equal(self.rmin, other.rmin)
            and np.array_equal(self.rmax, other.rmax)
            and self.unit == other.unit
            and self.rweight == other.rweight
            and self.rbin_num == other.rbin_num
        )

    @classmethod
    def from_dict(cls, the_dict):
        cls._check_dict(the_dict)
        parsed = cls._parse_params(the_dict)

        try:
            scales = new_scales(
                parsed.pop("rmin"),
                parsed.pop("rmax"),
                unit=parsed.pop("unit"),
            )
        except Exception as err:
            raise ConfigError(str(err)) from err
        return cls(scales, **parsed)

    @classmethod
    def create(
        cls,
        *,
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        unit: Unit = Unit.kpc,
        rweight: float | None = None,
        resolution: int | None = None,
    ) -> ScalesConfig:
        """
        Create a new instance with the given parameters.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in given unit of scales.
            rmax:
                Single or multiple upper scale limits in given unit of scales.
            unit:
                String describing the angular, physical, or comoving unit of
                correlation scales, see :obj:`~yaw.options.Unit` for valid
                options (default: kpc).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.

        Returns:
            New configuration instance.
        """
        the_dict = dict(
            rmin=rmin,
            rmax=rmax,
            unit=unit,
            rweight=rweight,
            resolution=resolution,
        )
        return cls.from_dict(the_dict)

    def modify(
        self,
        *,
        rmin: Iterable[float] | float | NotSet = NotSet,
        rmax: Iterable[float] | float | NotSet = NotSet,
        unit: Unit | NotSet = NotSet,
        rweight: float | None | NotSet = NotSet,
        resolution: int | None | NotSet = NotSet,
    ) -> ScalesConfig:
        """
        Create a new configuration instance with updated parameter values.

        Parameter values are only updated if they are provided as inputs to this
        function, otherwise they are retained from the original instance.

        Keyword Args:
            rmin:
                Single or multiple lower scale limits in given unit of scales.
            rmax:
                Single or multiple upper scale limits in given unit of scales.
            unit:
                String describing the angular, physical, or comoving unit of
                correlation scales, see :obj:`~yaw.options.Unit` for valid
                options (default: kpc).
            rweight:
                Optional power-law exponent :math:`\\alpha` used to weight pairs
                by their separation.
            resolution:
                Optional number of radial logarithmic bin used to approximate
                the weighting by separation.

        Returns:
            New instance with updated parameter values.
        """
        the_dict = self.to_dict()
        updates = dict(
            rmin=rmin,
            rmax=rmax,
            unit=unit,
            rweight=rweight,
            resolution=resolution,
        )
        the_dict.update(kv for kv in updates.items() if kv[1] is not NotSet)
        return self.from_dict(the_dict)


@dataclass(frozen=True)
class BinningConfig(YawConfig):
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

    _paramspec = (
        Parameter(
            name="zmin",
            help="Lowest redshift bin edge to generate (alternatively use 'edges').",
            type=float,
            default=None,
        ),
        Parameter(
            name="zmax",
            help="Highest redshift bin edge to generate (alternatively use 'edges').",
            type=float,
            default=None,
        ),
        Parameter(
            name="num_bins",
            help="Number of redshift bins to generate between 'zmin' and 'zmax'.",
            type=int,
            default=30,
        ),
        Parameter(
            name="method",
            help="Method used to compute the spacing of bin edges.",
            type=str,
            choices=BinMethod,
            default=BinMethod.linear,
            to_builtin=str,
        ),
        SequenceParameter(
            name="edges",
            help="Use these custom bin edges instead of generating them.",
            type=float,
            default=None,
        ),
        Parameter(
            name="closed",
            help="String indicating the side of the bin intervals that are closed.",
            type=str,
            choices=Closed,
            default=Closed.right,
            to_builtin=str,
        ),
    )

    binning: Binning
    """Container for the redshift bins."""
    method: BinMethod
    """Method used to compute the spacing of bin edges."""

    @property
    def edges(self) -> NDArray:
        """Array of redshift bin edges."""
        return self.binning.edges.tolist()

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
        """String indicating the side of the bin intervals that are closed."""
        return str(self.binning.closed)

    @property
    def is_custom(self) -> bool:
        """Whether the bin edges are provided by the user."""
        return self.method == "custom"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.method == other.method and self.binning == other.binning

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
        cls._check_dict(the_dict)
        parsed = cls._parse_params(the_dict)

        if parsed["edges"] is not None:
            method = BinMethod.custom
            try:
                binning = Binning(parsed["edges"], closed=parsed["closed"])
            except Exception as err:
                raise ConfigError(str(err)) from err

        elif parsed["zmin"] is None or parsed["zmax"] is None:
            raise ConfigError("either 'edges' or 'zmin' and 'zmax' are required")

        else:
            method = parsed["method"]
            try:
                binning = RedshiftBinningFactory(cosmology).get_method(method)(
                    parsed["zmin"],
                    parsed["zmax"],
                    parsed["num_bins"],
                    closed=parsed["closed"],
                )
            except Exception as err:
                raise ConfigError(str(err)) from err

        return cls(binning=binning, method=method)

    def to_dict(self) -> dict[str, Any]:
        if self.is_custom:
            return self._serialise(["method", "edges", "closed"])
        return self._serialise(["zmin", "zmax", "num_bins", "method", "closed"])

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
        cosmology: TypeCosmology | None = None,
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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
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
        the_dict = dict(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
        )
        return cls.from_dict(the_dict, cosmology=cosmology)

    def modify(
        self,
        *,
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = NotSet,
        method: BinMethod | str | NotSet = NotSet,
        edges: Iterable[float] | NotSet = NotSet,
        closed: Closed | str | NotSet = NotSet,
        cosmology: TypeCosmology | None | NotSet = NotSet,
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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            New instance with updated redshift bins.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        the_dict = self.to_dict()
        updates = dict(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
        )
        the_dict.update(kv for kv in updates.items() if kv[1] is not NotSet)
        return self.from_dict(the_dict, cosmology=cosmology)


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


default_cosmology = get_default_cosmology().name


@dataclass(frozen=True)
class Configuration(YawConfig):
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

    _paramspec = (
        ConfigSection(
            ScalesConfig,
            name="scales",
            help="Configuration of correlation measurement scales.",
            required=True,
        ),
        ConfigSection(
            BinningConfig,
            name="binning",
            help="Configuration of redshift binning for correlation measurements.",
            required=True,
        ),
        Parameter(
            name="cosmology",
            help="Cosmological model to use for distance computations.",
            type=CustomCosmology,
            default=default_cosmology,
            to_type=parse_cosmology,
            to_builtin=cosmology_to_yaml,
        ),
        Parameter(
            name="max_workers",
            help="Limit the number of workers for parallel operations (only multiprocessing).",
            type=int,
            default=None,
        ),
    )

    scales: ScalesConfig
    """Configuration of correlation scales."""
    binning: BinningConfig
    """Configuration of redshift bins used for sampling the redshift estimate."""
    cosmology: TypeCosmology | str
    """Cosmological model to use for distance computations."""
    max_workers: int | None
    """Limit the number of workers for parallel operations (all by default)."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            self.binning == other.binning
            and self.scales == other.scales
            and cosmology_is_equal(self.cosmology, other.cosmology)
            and self.max_workers == other.max_workers
        )

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]) -> Configuration:
        cls._check_dict(the_dict)

        with raise_configerror_with_level("scales"):
            scales = ScalesConfig.from_dict(the_dict["scales"])
        with raise_configerror_with_level("binning"):
            binning = BinningConfig.from_dict(the_dict["binning"])

        parsed = cls._parse_params(the_dict)
        return cls(scales, binning, **parsed)

    @classmethod
    @parallel.broadcasted
    def from_file(cls, path: Path | str) -> Configuration:
        logger.info("reading configuration file: %s", path)
        return super().from_file(path)

    @parallel.broadcasted
    def to_file(self, path: Path | str) -> None:
        logger.info("writing configuration file: %s", path)
        super().to_file(path)

    @classmethod
    def create(
        cls,
        *,
        # ScalesConfig
        rmin: Iterable[float] | float,
        rmax: Iterable[float] | float,
        unit: Unit = Unit.kpc,
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
            unit:
                String describing the angular, physical, or comoving unit of
                correlation scales, see :obj:`~yaw.options.Unit` for valid
                options (default: kpc).
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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
            cosmology:
                Optional cosmological model to use for distance computations.
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default, only multiprocessing).

        Returns:
            New configuration instance.

        .. note::
            Although the function parameters are optional, either ``zmin`` and
            ``zmax`` (generate bin edges), or ``edges`` (custom bin edges) must
            be provided.
        """
        the_dict = dict(
            scales=dict(
                rmin=rmin,
                rmax=rmax,
                unit=unit,
                rweight=rweight,
                resolution=resolution,
            ),
            binning=dict(
                zmin=zmin,
                zmax=zmax,
                num_bins=num_bins,
                method=method,
                edges=edges,
                closed=closed,
            ),
            cosmology=cosmology,
            max_workers=max_workers,
        )
        return cls.from_dict(the_dict)

    def modify(
        self,
        *,
        # ScalesConfig
        rmin: Iterable[float] | float | NotSet = NotSet,
        rmax: Iterable[float] | float | NotSet = NotSet,
        unit: Unit | NotSet = NotSet,
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
            unit:
                String describing the angular, physical, or comoving unit of
                correlation scales, see :obj:`~yaw.options.Unit` for valid
                options (default: kpc).
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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
            cosmology:
                Optional cosmological model to use for distance computations.
            max_workers:
                Limit the number of parallel workers for this operation (all by
                default, only multiprocessing).

        Returns:
            New instance with updated parameter values.
        """
        the_dict = self.to_dict()

        # scales parameters
        updates = dict(
            rmin=rmin,
            rmax=rmax,
            unit=unit,
            rweight=rweight,
            resolution=resolution,
        )
        the_dict["scales"].update(kv for kv in updates.items() if kv[1] is not NotSet)

        # binning parameters
        updates = dict(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
        )
        the_dict["binning"].update(kv for kv in updates.items() if kv[1] is not NotSet)

        # self-owned parameters
        updates = dict(
            cosmology=cosmology,
            max_workers=max_workers,
        )
        the_dict.update(kv for kv in updates.items() if kv[1] is not NotSet)

        return self.from_dict(the_dict)
