from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.config import DEFAULT, OPTIONS
from yaw.config.abc import BaseConfig
from yaw.config.utils import ConfigError
from yaw.core.cosmology import BinFactory, TypeCosmology
from yaw.core.docs import Parameter
from yaw.core.math import array_equal

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["BinningConfig"]


@dataclass(frozen=True)
class BinningConfig(BaseConfig):
    """TODO"""

    zbins: NDArray[np.float_] = field(
        metadata=Parameter(
            type=float,
            nargs="*",
            help="list of custom redshift bin edges, if method is set to 'manual'",
        )
    )
    """Edges of redshift bins."""
    method: str = field(
        default=DEFAULT.Binning.method,
        metadata=Parameter(
            choices=OPTIONS.binning,
            help="redshift binning method, 'logspace' means equal size in log(1+z)",
            default_text="(default: %(default)s)",
        ),
    )
    """Method used to create redshift binning (defaults to 'manual' if `zbins`
    are provided), see :obj:`~yaw.config.options.Options.binning` for options."""
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
        if len(self.zbins) < 2:
            raise ConfigError("'zbins' must have at least two edges")
        object.__setattr__(self, "zbins", np.asarray(self.zbins))
        BinFactory.check(self.zbins)
        object.__setattr__(self, "zmin", float(self.zbins[0]))
        object.__setattr__(self, "zmax", float(self.zbins[-1]))
        object.__setattr__(self, "zbin_num", len(self.zbins) - 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.method != other.method or not array_equal(self.zbins, other.zbins):
            return False
        return True

    def __repr__(self) -> str:
        name = self.__class__.__name__
        zbin_num = self.zbin_num
        z = f"{self.zmin:.3f}...{self.zmax:.3f}"
        method = self.method
        return f"{name}({zbin_num=}, {z=}, {method=})"

    @property
    def is_manual(self) -> bool:
        """Whether the redshift bins are set manually."""
        return self.method == "manual"

    @classmethod
    def create(
        cls,
        *,
        zbins: NDArray[np.float_] | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zbin_num: int = DEFAULT.Binning.zbin_num,
        method: str = DEFAULT.Binning.method,
        cosmology: TypeCosmology | str | None = None,
    ) -> BinningConfig:
        """Create a new redshift binning configuration.

        If redshift bins (``zbins``) are provided, ``method`` is set to
        ``"manual"`` and the bins edges are used. If ``zmin`` and ``zmax`` are
        provided, instead generates a specified number of bins between this
        lower and upper redshift limit. The spacing of the bins depends the
        generation method, the default is a linear spacing.

        .. Note::

            The ``cosmology`` parameter is only used when generating binnings
            that require cosmological distance computations.

        Args:
            zbins (:obj:`NDArray[np.float_]`):
                Monotonically increasing redshift bin edges, including the upper
                edge (ignored if ``zmin`` and ``zmax`` are provided).
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
            :obj:`BinningConfig`
        """
        auto_args_set = (zmin is not None, zmax is not None)
        manual_args_set = (zbins is not None,)
        if not all(manual_args_set) and not all(auto_args_set):
            raise ConfigError("either 'zbins' or 'zmin' and 'zmax' are required")

        elif all(auto_args_set):  # generate zbins
            if all(manual_args_set):
                warnings.warn(
                    "'zbins' set but ignored since 'zmin' and 'zmax' are provided"
                )
            if not isinstance(method, str):
                raise ValueError("'method' must a string")
            zbins = BinFactory(zmin, zmax, zbin_num, cosmology).get(method)

        else:  # use provided zbins
            method = "manual"

        return cls(zbins=zbins, method=method)

    def modify(
        self,
        zbins: NDArray[np.float_] | None = DEFAULT.NotSet,
        zmin: float | None = DEFAULT.NotSet,
        zmax: float | None = DEFAULT.NotSet,
        zbin_num: int = DEFAULT.NotSet,
        method: str = DEFAULT.NotSet,
        cosmology: TypeCosmology | str | None = None,
    ) -> BinningConfig:
        """Create a copy of the current configuration with updated parameter
        values.

        The method arguments are identical to :meth:`create`. Values that should
        not be modified are by default represented by the special value
        :obj:`~yaw.config.default.NotSet`.

        .. Note::

            The ``cosmology`` parameter is only used when generating binnings
            that require cosmological distance computations.
        """
        if zbins is not DEFAULT.NotSet:
            kwargs = dict(zbins=zbins)
        else:
            inputs = dict(zmin=zmin, zmax=zmax, zbin_num=zbin_num, method=method)
            mods = {k: v for k, v in inputs.items() if v is not DEFAULT.NotSet}
            if self.is_manual:  # use every input as parameter
                kwargs = mods
            else:  # keep the existing parameters and update with inputs
                kwargs = self.to_dict()
                kwargs.update(mods)
        return self.__class__.create(**kwargs, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        if self.is_manual:
            return dict(method=self.method, zbins=self.zbins.tolist())
        else:
            return dict(
                zmin=self.zmin,
                zmax=self.zmax,
                zbin_num=self.zbin_num,
                method=self.method,
            )

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, Any], cosmology: TypeCosmology | None = None
    ) -> BinningConfig:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`~yaw.core.cosmology.CustomCosmology`, optional):
                Cosmological model used for distance calculations. For a custom
                model, refer to :mod:`yaw.core.cosmology`.
        """
        return cls.create(**the_dict, cosmology=cosmology)
