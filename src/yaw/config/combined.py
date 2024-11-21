"""
Implements the central configuration class for `yet_another_wizz`. This stores
the configuration of measurement scales, redshift binning and cosmological model
to use for correlation fuction measurements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, get_args

import astropy.cosmology

from yaw.config.base import BaseConfig, ConfigError, Immutable, Parameter, ParamSpec
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

logger = logging.getLogger(__name__)


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
                correlation scales (default: kpc).
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
        cosmology = parse_cosmology(cosmology)

        scales = ScalesConfig.create(
            rmin=rmin, rmax=rmax, unit=unit, rweight=rweight, resolution=resolution
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

        return cls(
            scales=scales, binning=binning, cosmology=cosmology, max_workers=max_workers
        )

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
                correlation scales (default: kpc).
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
            rmin=rmin, rmax=rmax, unit=unit, rweight=rweight, resolution=resolution
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
