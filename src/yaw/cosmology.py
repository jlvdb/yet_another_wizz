"""
Implements some utilities for cosmological distance computations.

Most notably a function to convert angular diameter distances to angles at a
given redshift and a given cosmological model. By default, uses named
cosmologies from ``astropy``, but also provides a base-class to implement custom
cosmological models. Note that the cosmology typically has a minor impact on the
derive clustering redshifts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units
from astropy.cosmology import FLRW, Planck15, cosmology_equal, z_at_value

from yaw.binning import Binning
from yaw.options import BinMethodAuto, Closed, Unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

TypeCosmology = Union[FLRW, "CustomCosmology"]  # used with get_args

__all__ = [
    "CustomCosmology",
    "cosmology_is_equal",
    "get_default_cosmology",
    "new_scales",
]


class CustomCosmology(ABC):
    """Meta-class that defines interface for custom cosmologies that are
    compatible with `yet_another_wizz` code."""

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        """
        Compute the comoving distance.

        Args:
            z:
                A single or an array of redshifts.

        Returns:
            Float or numpy array with comoving distance in Mpc for given input
            redshifts.
        """
        pass

    @abstractmethod
    def angular_diameter_distance(self, z: ArrayLike) -> ArrayLike:
        """
        Compute the angular diameter distance.

        Args:
            z:
                A single or an array of redshifts.

        Returns:
            Float or numpy array with angular diameter distance in Mpc for given
            input redshifts.
        """
        pass


def cosmology_is_equal(cosmo1: TypeCosmology, cosmo2: TypeCosmology) -> bool:
    """Compare if two ``astropy`` cosmologies are equal. Always ``True`` for
    instances of ``CustomCosmology``."""
    if not isinstance(cosmo1, (FLRW, CustomCosmology)):
        raise TypeError("'cosmo1' is not a valid cosmology type")
    if not isinstance(cosmo2, (FLRW, CustomCosmology)):
        raise TypeError("'cosmo2' is not a valid cosmology type")

    is_custom_1 = isinstance(cosmo1, CustomCosmology)
    is_custom_2 = isinstance(cosmo2, CustomCosmology)

    if is_custom_1 and is_custom_2:
        return True

    elif not is_custom_1 and not is_custom_2:
        return cosmology_equal(cosmo1, cosmo2)

    return False


def get_default_cosmology() -> FLRW:
    return Planck15


class Scales(ABC):
    """
    Container for correlation scales in angular, physical or comoving units.

    Args:
        scales_min:
            Single or multiple lower scale limits in given unit of scales.
        scale_max:
            Single or multiple upper scale limits in given unit of scales.

    Keyword Args:
        unit:
            String describing the angular, physical, or comoving unit of
            correlation scales (default: kpc).
    """

    def _set_scales(self, scale_min: ArrayLike, scale_max: ArrayLike) -> None:
        scale_min: NDArray = np.atleast_1d(scale_min)
        scale_max: NDArray = np.atleast_1d(scale_max)

        if scale_min.ndim != scale_max.ndim and scale_min.ndim != 1:
            raise ValueError(  # TODO: ConfigError
                "min/max scales must be scalars or one-dimensional arrays"
            )
        if len(scale_min) != len(scale_max):
            # TODO: ConfigError
            raise ValueError("number of elements in min and max scales does not match")
        if np.any((scale_max - scale_min) <= 0.0):
            raise ValueError(  # TODO: ConfigError
                "all min scales must be smaller than corresponding max scales"
            )

        self.scale_min = scale_min
        self.scale_max = scale_max

    def __repr__(self) -> str:
        min = self.scale_min.tolist()
        max = self.scale_max.tolist()
        return f"{type(self).__name__}({min=}, {max=}, unit='{self.unit}')"

    @property
    def num_scales(self) -> int:
        """Number of scale limits."""
        return len(self.scale_min)

    @abstractmethod
    def _compute_angle(
        self, scales: NDArray, redshift: float, cosmology: TypeCosmology
    ) -> NDArray:
        pass

    def get_angle_radian(
        self, redshift: float, cosmology: TypeCosmology | None = None
    ) -> tuple[NDArray, NDArray]:
        """
        Convert the scale limits to angles in radian for a given cosmological
        model and redshift.

        Args:
            redshift:
                Redshift at which the scales are converted.
            cosmology:
                Optional cosmological model used for distance conversions.
        """
        cosmology = cosmology or get_default_cosmology()
        return (
            self._compute_angle(self.scale_min, redshift, cosmology),
            self._compute_angle(self.scale_max, redshift, cosmology),
        )


def new_scales(
    scale_min: ArrayLike, scale_max: ArrayLike, *, unit: Unit = Unit.kpc
) -> Scales:
    """
    Create a new container for correlation scales in angular, physical or
    comoving units.

    Args:
        scales_min:
            Single or multiple lower scale limits in given unit of scales.
        scale_max:
            Single or multiple upper scale limits in given unit of scales.

    Keyword Args:
        unit:
            String describing the angular, physical, or comoving unit of
            correlation scales (default: kpc).
    """
    unit = Unit(unit)

    if unit in (Unit.rad, Unit.deg, Unit.arcmin, Unit.arcsec):
        scales_cls = AngularScales
    elif unit in (Unit.kpc, Unit.Mpc):
        scales_cls = PhysicalScales
    else:
        scales_cls = ComovingScales

    return scales_cls(scale_min, scale_max, unit=unit)


class AngularScales(Scales):
    def __init__(
        self,
        scale_min: ArrayLike,
        scale_max: ArrayLike,
        *,
        unit: Unit,
    ) -> None:
        self.unit = Unit(unit)
        if self.unit not in (Unit.rad, Unit.deg, Unit.arcmin, Unit.arcsec):
            raise ValueError(f"'{unit}' is not a valid angular separation unit")

        self._set_scales(scale_min, scale_max)

    def _compute_angle(
        self, scales: NDArray, redshift: float, cosmology: TypeCosmology
    ) -> NDArray:
        if self.unit == Unit.rad:
            return scales

        if self.unit == Unit.arcsec:
            scales = scales / 3600.0
        elif self.unit == Unit.arcsec:
            scales = scales / 60.0
        return np.deg2rad(scales)


class PhysicalScales(Scales):
    def __init__(
        self,
        scale_min: ArrayLike,
        scale_max: ArrayLike,
        *,
        unit: Unit,
    ) -> None:
        self.unit = Unit(unit)
        if self.unit not in (Unit.kpc, Unit.Mpc):
            raise ValueError(f"'{unit}' is not a valid physical separation unit")

        self._set_scales(scale_min, scale_max)

    def _compute_angle(
        self, scales: NDArray, redshift: float, cosmology: TypeCosmology
    ) -> NDArray:
        if self.unit == Unit.kpc:
            scales = scales / 1000.0

        ang_diam_dist_mpc = cosmology.angular_diameter_distance(redshift)
        if isinstance(ang_diam_dist_mpc, units.Quantity):
            ang_diam_dist_mpc = ang_diam_dist_mpc.value
        return scales / ang_diam_dist_mpc


class ComovingScales(Scales):
    def __init__(
        self,
        scale_min: ArrayLike,
        scale_max: ArrayLike,
        *,
        unit: Unit,
    ) -> None:
        self.unit = Unit(unit)
        if self.unit not in (Unit.kpc_h, Unit.Mpc_h):
            raise ValueError(f"'{unit}' is not a valid comoving separation unit")

        self._set_scales(scale_min, scale_max)

    def _compute_angle(
        self, scales: NDArray, redshift: float, cosmology: TypeCosmology
    ) -> NDArray:
        if self.unit == Unit.kpc_h:
            scales = scales / 1000.0

        comov_dist_mpc = cosmology.comoving_distance(redshift)
        if isinstance(comov_dist_mpc, units.Quantity):
            comov_dist_mpc = comov_dist_mpc.value
        return scales / comov_dist_mpc


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

        edges = z_at_value(self.cosmology.comoving_distance, comov_edges)
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
