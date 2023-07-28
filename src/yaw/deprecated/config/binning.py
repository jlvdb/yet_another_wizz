from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from deprecated import deprecated

from yaw.config import OPTIONS
from yaw.config import default as DEFAULT
from yaw.config.utils import ConfigError
from yaw.core.abc import DictRepresentation
from yaw.core.cosmology import BinFactory, TypeCosmology
from yaw.core.docs import Parameter
from yaw.core.math import array_equal

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["AutoBinningConfig", "ManualBinningConfig", "make_binning_config"]


class BaseBinningConfig(DictRepresentation):
    """Base class for redshift binning configuration."""

    zbins: NDArray[np.float_]
    """Edges of redshift bins."""
    method: str
    """Method used to create redshift binning, ``manual`` or either of
    :obj:`~yaw.config.options.Options.binning`."""

    def __repr__(self) -> str:
        name = self.__class__.__name__
        zbin_num = self.zbin_num
        z = f"{self.zmin:.3f}...{self.zmax:.3f}"
        method = self.method
        return f"{name}({zbin_num=}, {z=}, {method=})"


@deprecated(
    reason="merged into yaw.config.BinningConfig", action="module", version="2.5.5"
)
@dataclass(frozen=True, repr=False)
class ManualBinningConfig(BaseBinningConfig):
    """Configuration that specifies a manual redshift binning.

    Args:
        zbins (:obj:`NDArray`):
            Edges of redshift bins, must increase monotonically.
    """

    zbins: NDArray[np.float_] = field(
        metadata=Parameter(
            type=float,
            nargs="*",
            help="list of custom redshift bin edges, if method is set to 'manual'",
        )
    )

    def __post_init__(self) -> None:
        if len(self.zbins) < 2:
            raise ConfigError("'zbins' must have at least two edges")
        BinFactory.check(self.zbins)
        object.__setattr__(self, "zbins", np.asarray(self.zbins))

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not array_equal(self.zbins, other.zbins):
            return False
        return True

    @property
    def method(self) -> str:
        """Method used to create redshift binning, always ``manual``."""
        return "manual"

    @property
    def zmin(self) -> float:
        """Lowest redshift bin edge."""
        return float(self.zbins[0])

    @property
    def zmax(self) -> float:
        """Highest redshift bin edge."""
        return float(self.zbins[-1])

    @property
    def zbin_num(self) -> int:
        """Number of redshift bins."""
        return len(self.zbins) - 1

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> ManualBinningConfig:
        new_dict = {key: val for key, val in the_dict.items()}
        zbins = new_dict.pop("zbins")
        return cls(np.asarray(zbins), **new_dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(method=self.method, zbins=self.zbins.tolist())


@deprecated(
    reason="merged into yaw.config.BinningConfig", action="module", version="2.5.5"
)
@dataclass(frozen=True, repr=False)
class AutoBinningConfig(BaseBinningConfig):
    """Configuration that generates a redshift binning.

    To generate a redshift binning use the :meth:`generate` method.

    Args:
        zbins (:obj:`NDArray`):
            Edges of redshift bins, must increase monotonically.
        method (:obj:`str`):
            Method used to create redshift binning, either of
            :obj:`~yaw.config.options.Options.binning`.
    """

    zbins: NDArray[np.float_]
    method: str = field(
        default=DEFAULT.Configuration.binning.method,
        metadata=Parameter(
            choices=OPTIONS.binning,
            help="redshift binning method, 'logspace' means equal size in log(1+z)",
            default_text="(default: %(default)s)",
        ),
    )
    """Method used to create redshift binning, see
    :obj:`~yaw.config.options.Options.binning`."""
    zmin: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="lower redshift limit",
            default_text="(default: %(default)s)",
        ),
    )
    """Lowest redshift bin edge."""
    zmax: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="upper redshift limit",
            default_text="(default: %(default)s)",
        ),
    )
    """Highest redshift bin edge."""
    zbin_num: int = field(
        default=DEFAULT.Binning.zbin_num,
        init=False,
        metadata=Parameter(
            type=int,
            help="number of redshift bins",
            default_text="(default: %(default)s)",
        ),
    )
    """Number of redshift bins."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "zmin", float(self.zbins[0]))
        object.__setattr__(self, "zmax", float(self.zbins[-1]))
        object.__setattr__(self, "zbin_num", len(self.zbins) - 1)

    @classmethod
    def generate(
        cls,
        zmin: float,
        zmax: float,
        zbin_num: int = DEFAULT.Binning.zbin_num,
        method: str = DEFAULT.Binning.method,
        cosmology: TypeCosmology | None = None,
    ) -> AutoBinningConfig:
        """Generate a new redshift binning configuration.

        Generates a specified number of bins between a lower and upper redshift
        limit. The spacing of the bins depends the generation method, the
        default is a linear spacing.

        Args:
            zmin (:obj:`float`):
                Minimum redshift, lowest redshift edge.
            zmax (:obj:`float`):
                Maximum redshift, highest redshift edge.
            zbin_num (:obj:`int`, optional):
                Number of redshift bins to generate.
            method (:obj:`str`, optional):
                Method used to create redshift binning, for a list of valid
                options and their description see
                :obj:`~yaw.config.options.Options.binning`.
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`~yaw.core.cosmology.CustomCosmology`, optional):
                Cosmological model used for distance calculations. For a custom
                model, refer to :mod:`yaw.core.cosmology`.

        Returns:
            :obj:`AutoBinningConfig`
        """
        if not isinstance(method, str):
            raise ValueError("'method' must a string")
        zbins = BinFactory(zmin, zmax, zbin_num, cosmology).get(method)
        return cls(zbins, method)

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if self.method != other.method:
            return False
        if not array_equal(self.zbins, other.zbins):
            return False
        return True

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, Any], cosmology: TypeCosmology | None = None
    ) -> AutoBinningConfig:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`~yaw.core.cosmology.CustomCosmology`, optional):
                Cosmological model used for distance calculations. For a custom
                model, refer to :mod:`yaw.core.cosmology`.
        """
        return cls.generate(**the_dict, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            zmin=self.zmin, zmax=self.zmax, zbin_num=self.zbin_num, method=self.method
        )


@deprecated(
    reason="use yaw.config.BinningConfig.create() instead",
    action="module",
    version="2.5.5",
)
def make_binning_config(
    *,
    cosmology: TypeCosmology | str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    zbin_num: int = DEFAULT.Binning.zbin_num,
    method: str = DEFAULT.Binning.method,
    zbins: NDArray[np.float_] | None = None,
) -> ManualBinningConfig | AutoBinningConfig:
    """
    Helper function to construct a binning configuration.

    The ``cosmology`` argument is always required. If redshift bins (``zbins``)
    are provided, a :obj:`ManualBinningConfig` is returned, otherwise an
    :obj:`AutoBinningConfig`. Issues a warning if any argument is set but
    ignored by the returned configuration instance.
    """
    auto_args_set = (zmin is not None, zmax is not None)
    manual_args_set = (zbins is not None,)
    if not all(manual_args_set) and not all(auto_args_set):
        raise ConfigError("either 'zbins' or 'zmin' and 'zmax' are required")
    elif all(auto_args_set):
        if all(manual_args_set):
            warnings.warn(
                "'zbins' set but ignored since 'zmin' and 'zmax' are provided"
            )
        return AutoBinningConfig.generate(
            zmin=zmin, zmax=zmax, zbin_num=zbin_num, method=method, cosmology=cosmology
        )
    else:
        return ManualBinningConfig(zbins)
