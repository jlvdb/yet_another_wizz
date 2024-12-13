"""
Implements the central configuration class for `yet_another_wizz`. This stores
the configuration of measurement scales, redshift binning and cosmological model
to use for correlation fuction measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

import astropy.cosmology

from yaw.config.base import (
    ConfigError,
    ConfigSection,
    Parameter,
    YawConfig,
    raise_configerror_with_level,
)
from yaw.config.binning import BinningConfig
from yaw.config.scales import ScalesConfig
from yaw.cosmology import (
    CustomCosmology,
    TypeCosmology,
    cosmology_is_equal,
    get_default_cosmology,
)
from yaw.options import BinMethod, Closed, NotSet, Unit
from yaw.utils import parallel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any

__all__ = [
    "Configuration",
]

logger = logging.getLogger("yaw.config")  # instead of "yaw.config.combined"


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
            help="hulp",
            required=True,
        ),
        ConfigSection(
            BinningConfig,
            name="binning",
            help="hulp",
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
            help="Limit the number of workers for parallel operations.",
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
                default).

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
                Limit the  number of parallel workers for this operation (all by
                default).

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
