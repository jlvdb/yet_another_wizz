from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.core import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.cosmology import BINNING_OPTIONS, BinFactory, TypeCosmology
from yaw.core.docs import Parameter
from yaw.core.math import array_equal

from yaw.config.scales import ScalesConfig
from yaw.config.utils import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class BaseBinningConfig(DictRepresentation):

    zbins: NDArray[np.float_]
    method: str

    def __repr__(self) -> str:
        name = self.__class__.__name__
        zbin_num = self.zbin_num
        z = f"{self.zmin:.3f}...{self.zmax:.3f}"
        method = self.method
        return f"{name}({zbin_num=}, {z=}, {method=})"


@dataclass(frozen=True, repr=False)
class ManualBinningConfig(BaseBinningConfig):

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
        return "manual"

    @property
    def zmin(self) -> float:
        return float(self.zbins[0])

    @property
    def zmax(self) -> float:
        return float(self.zbins[-1])

    @property
    def zbin_num(self) -> int:
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

    zbins: NDArray[np.float_]
    method: str = field(
        default=DEFAULT.Config.binning.method,
        metadata=Parameter(
            choices=BINNING_OPTIONS,
            help="redshift binning method, 'logspace' means equal size in "
                 "log(1+z)",
            default_text="(default: %(default)s)"))
    zmin: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="lower redshift limit",
            default_text="(default: %(default)s)"))
    zmax: float = field(
        init=False,
        metadata=Parameter(
            type=float,
            help="upper redshift limit",
            default_text="(default: %(default)s)"))
    zbin_num: int = field(
        default=DEFAULT.AutoBinning.zbin_num,
        init=False,
        metadata=Parameter(
            type=int,
            help="number of redshift bins",
            default_text="(default: %(default)s)"))

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
    ) -> ScalesConfig:
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
    auto_args_set =  (zmin is not None, zmax is not None, zbin_num is not None)
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
