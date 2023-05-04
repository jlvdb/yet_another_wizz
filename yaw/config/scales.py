from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.core import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.docs import Parameter
from yaw.core.math import array_equal
from yaw.core.utils import scales_to_keys

from yaw.config.utils import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


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
        if (
            isinstance(self.rmin, (Sequence, np.ndarray)) and
            isinstance(self.rmax, (Sequence, np.ndarray))
        ):
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
