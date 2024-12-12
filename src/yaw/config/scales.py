"""
Implements a class that stores the configuration of angular scales used for
correlation function measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from yaw.config.base import Parameter, YawConfig
from yaw.cosmology import Scales, new_scales
from yaw.options import NotSet, Unit

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ScalesConfig",
]


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
        Parameter(
            name="rmin",
            help="Single or multiple lower scale limits in given 'unit'.",
            type=float,
            is_sequence=True,
        ),
        Parameter(
            name="rmax",
            help="Single or multiple upper scale limits in given 'unit'.",
            type=float,
            is_sequence=True,
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
            nullable=True,
        ),
        Parameter(
            name="resolution",
            help="Number of radial logarithmic bin used to approximate the weighting by separation.",
            type=int,
            default=None,
            nullable=True,
        ),
    )

    scales: Scales
    """Correlation scales in angular or physical units."""
    rweight: float | None
    """Optional power-law exponent :math:`\\alpha` used to weight pairs by their
    separation."""
    resolution: int | None
    """Optional number of radial logarithmic bin used to approximate the
    weighting by separation."""

    @property
    def rmin(self) -> float | list[float]:
        """Single or multiple lower scale limits in given unit of scales."""
        return self.scales.scale_min.squeeze().tolist()

    @property
    def rmax(self) -> float | list[float]:
        """Single or multiple upper scale limits in given unit of scales."""
        return self.scales.scale_max.squeeze().tolist()

    @property
    def unit(self) -> str:
        """String describing the angular, physical, or comoving unit of
        correlation scales (default: kpc)."""
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

        scales = new_scales(
            parsed.pop("rmin"),
            parsed.pop("rmax"),
            unit=parsed.pop("unit"),
        )
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
