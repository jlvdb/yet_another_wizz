from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from deprecated import deprecated

from yaw.config import default as DEFAULT
from yaw.config.abc import BaseConfig
from yaw.config.utils import ConfigError
from yaw.core.cosmology import Scale
from yaw.core.docs import Parameter
from yaw.core.math import array_equal

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["ScalesConfig"]


@dataclass(frozen=True)
class ScalesConfig(BaseConfig):
    """Configuration of scales used for correlation measurements.

    Correlation functions are measured on one or many intervals
    :math:`r_{\\rm min} \\leq r < r_{\\rm max}` angular diameter distance in
    kpc. When measuring correlations, this scale is coverted to angles at the
    current redshift.

    Additionally, pairs can be weighted by their separation
    :math:`r^\\alpha` if a power-law exponent is provided through ``rweight``.
    The weighting is applied logarithmically spaced bins of separation (based
    on the logarithmic bin centers). This is an approximation to actually
    weighting each pair individually and the resolution of this approximation
    can be controlled by setting the number of bins.

    Args:
        rmin (:obj:`float`, :obj:`list[float]`):
            Single or multiple lower scale limits in kpc (angular diameter
            distance).
        rmax (:obj:`float`, :obj:`list[float]`):
            Single or multiple upper scale limits in kpc (angular diameter
            distance).
        rweight (:obj:`float`, optional):
            Power-law exponent used to weight pairs by their separation.
        rbin_num (:obj:`int`, optional):
            Number of radial logarithmic bin used to approximate the weighting
            by separation.
    """

    rmin: list[float] | float = field(
        metadata=Parameter(
            type=float,
            nargs="*",
            required=True,
            help="(list of) lower scale limit in kpc (pyhsical)",
        )
    )
    """Lower scale limit(s) in kpc (angular diameter distance)."""
    rmax: list[float] | float = field(
        metadata=Parameter(
            type=float,
            nargs="*",
            required=True,
            help="(list of) upper scale limit in kpc (pyhsical)",
        )
    )
    """Upper scale limit(s) in kpc (angular diameter distance)."""
    rweight: float | None = field(
        default=DEFAULT.Scales.rweight,
        metadata=Parameter(
            type=float,
            help="weight galaxy pairs by their separation to power 'rweight'",
            default_text="(default: no weighting applied)",
        ),
    )
    """Power-law exponent used to weight pairs by their separation."""
    rbin_num: int = field(
        default=DEFAULT.Scales.rbin_num,
        metadata=Parameter(
            type=int,
            help="number of bins in log r used (i.e. resolution) to compute distance weights",
            default_text="(default: %(default)s)",
        ),
    )
    """Number of radial logarithmic bin used to approximate the weighting by
    separation."""

    def __post_init__(self) -> None:
        msg_scale_error = "scales violates 'rmin' < 'rmax'"
        # validation, set to basic python types
        scalars = (float, int, np.number)
        if isinstance(self.rmin, (Sequence, np.ndarray)) and isinstance(
            self.rmax, (Sequence, np.ndarray)
        ):
            if len(self.rmin) != len(self.rmax):
                raise ConfigError(
                    "number of elements in 'rmin' and 'rmax' do not match"
                )
            # for clean YAML conversion
            for rmin, rmax in zip(self.rmin, self.rmax):
                if rmin >= rmax:
                    raise ConfigError(msg_scale_error)
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
                raise ConfigError(msg_scale_error)
        else:
            raise ConfigError("'rmin' and 'rmax' must be both sequences or float")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                array_equal(self.as_array(), other.as_array())
                and self.rweight == other.rweight
                and self.rbin_num == other.rbin_num
            )
        return NotImplemented

    def __getitem__(self, idx: int) -> Scale:
        scales = self.as_array()
        return Scale(rmin=scales[idx, 0], rmax=scales[idx, 1])

    def __iter__(self) -> Iterator[Scale]:
        for rmin, rmax in self.as_array():
            yield Scale(rmin=rmin, rmax=rmax)

    def modify(
        self,
        rmin: list[float] | float = DEFAULT.NotSet,
        rmax: list[float] | float = DEFAULT.NotSet,
        rweight: float | None = DEFAULT.NotSet,
        rbin_num: int = DEFAULT.NotSet,
    ) -> ScalesConfig:
        return super().modify(rmin=rmin, rmax=rmax, rweight=rweight, rbin_num=rbin_num)

    def as_array(self) -> NDArray[np.float_]:
        """Obtain the scales cuts as array of shape (2, N)"""
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))

    @deprecated("use [str(scale) for scale in ScalesConfig] instead", version="2.3.1")
    def dict_keys(self) -> list[str]:
        """Get the scale cuts formatted as a list of strings.

        Format is ``kpc[rmin]t[rmax]``, used as keys to pack outputs of
        correlation measurements in a dictionary when measuring with multiple
        scales cuts.

        .. deprecated:: 2.3.1
            Use instead

            >>> [str(scale) for scale in ScalesConfig]
            ...
        """
        return [str(scale) for scale in self]  # pragma: no cover
