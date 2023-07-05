from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.core.abc import DictRepresentation
from yaw.core.cosmology import BinFactory, TypeCosmology
from yaw.core.docs import Parameter
from yaw.core.math import array_equal

from yaw.config import default as DEFAULT, OPTIONS
from yaw.config.utils import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, repr=False)
class ManualBinningConfig(BaseBinningConfig):
    """Configuration that specifies a manual redshift binning.

    Args:
        zbins (:obj:`NDArray`):
            Edges of redshift bins, must increase monotonically.
    """

    zbins: NDArray[np.float_] = field(
        metadata=Parameter(
            type=float, nargs="*",
            help="list of custom redshift bin edges, if provided, other "
                 "binning parameters are omitted, method is set to 'manual'"))

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
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        **kwargs
    ) -> ManualBinningConfig:
        return cls(np.asarray(the_dict["zbins"]))

    def to_dict(self) -> dict[str, Any]:
        return dict(method=self.method, zbins=self.zbins.tolist())


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
            help="redshift binning method, 'logspace' means equal size in "
                 "log(1+z)",
            default_text="(default: %(default)s)"))
    """Method used to create redshift binning, see
    :obj:`~yaw.config.options.Options.binning`."""
    zmin: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="lower redshift limit",
            default_text="(default: %(default)s)"))
    """Lowest redshift bin edge."""
    zmax: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="upper redshift limit",
            default_text="(default: %(default)s)"))
    """Highest redshift bin edge."""
    zbin_num: int = field(
        default=DEFAULT.AutoBinning.zbin_num,
        init=False,
        metadata=Parameter(
            type=int,
            help="number of redshift bins",
            default_text="(default: %(default)s)"))
    """Number of redshift bins."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "zmin", float(self.zbins[0]))
        object.__setattr__(self, "zmax", float(self.zbins[-1]))
        object.__setattr__(self, "zbin_num", len(self.zbins)-1)

    @classmethod
    def generate(
        cls,
        zmin: float,
        zmax: float,
        zbin_num: int = DEFAULT.AutoBinning.zbin_num,
        method: str = DEFAULT.AutoBinning.method,
        cosmology: TypeCosmology | None = None
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
        zbins = BinFactory(zmin, zmax, zbin_num, cosmology).get(method)
        return cls(zbins, method)

    def __eq__(self, other: ManualBinningConfig) -> bool:
        if not array_equal(self.zbins, other.zbins):
            return False
        if self.method != other.method:
            return False
        return True

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        cosmology: TypeCosmology | None = None
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
            zmin=self.zmin,
            zmax=self.zmax,
            zbin_num=self.zbin_num,
            method=self.method)


def warn_binning_args_ignored(
    zmin: bool | float | None,
    zmax: bool | float | None,
    zbin_num: bool | int | None
) -> None:
    """Issue a warning that values are ignored, if any of the arguments is
    set."""
    # NOTE: NotSet is also False
    if zmin or zmax or zbin_num:
        logger.warn(
            "'zmin', 'zmax', 'nbins' are ignored if 'zbins' is provided")


def make_binning_config(
    cosmology: TypeCosmology | str | None,
    zmin: float | None = None,
    zmax: float | None = None,
    zbin_num: int | None = None,
    method: str | None = None,
    zbins: NDArray[np.float_] | None = None,
) -> ManualBinningConfig | AutoBinningConfig:
    """
    Helper function to construct a binning configuration.

    The ``cosmology`` argument is always required. If redshift bins (``zbins``)
    are provided, a :obj:`ManualBinningConfig` is returned, otherwise an
    :obj:`AutoBinningConfig`. Issues a warning if any argument is set but
    ignored by the returned configuration instance.
    """
    auto_args_set = (zmin is not None, zmax is not None, zbin_num is not None)
    if zbins is None and not all(auto_args_set):
        raise ConfigError(
            "either 'zbins' or 'zmin', 'zmax', 'zbin_num' are required")
    elif all(auto_args_set):
        return AutoBinningConfig.generate(
            zmin=zmin, zmax=zmax, zbin_num=zbin_num,
            method=method, cosmology=cosmology)
    else:
        warn_binning_args_ignored(*auto_args_set)
        return ManualBinningConfig(zbins)
