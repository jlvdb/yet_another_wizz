"""
Implements a class that stores the configuration redshift bins used to split the
catalogs when measuring angular correlations.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from yaw.binning import Binning
from yaw.config.base import BaseConfig, ConfigError, Immutable, Parameter, ParamSpec
from yaw.cosmology import RedshiftBinningFactory, TypeCosmology
from yaw.options import BinMethod, Closed, NotSet, get_options

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import NDArray

__all__ = [
    "BinningConfig",
]


class BinningConfig(BaseConfig, Immutable):
    """
    Configuration of the redshift binning for correlation function measurements.

    Correlations are measured in bins of redshift, which determines the
    redshift-resolution of the clustering redshift estimate. This configuration
    class offers three automatic methods to generate these bins between a
    minimum and maximum redshift:

    - ``linear`` (default): bin edges spaced linearly in redshift :math:`z`,
    - ``comoving``: bin edges spaced linearly in comoving distance
      :math:`\\chi(z)`, and
    - ``logspace``: bin edges spaced linearly in :math:`1+\\ln(z)`.

    Alternatively, custom bin edges may be provided.

    .. note::
        The preferred way to create a new configuration instance is using the
        :meth:`create()` constructor.

        All configuration objects are immutable. To modify an existing
        configuration, create a new instance with updated values by using the
        :meth:`modify()` method. The bin edges are recomputed when necessary.
    """

    binning: Binning
    """Container for the redshift bins."""
    method: BinMethod
    """Method used to generate the bin edges, must be either of ``linear``,
    ``comoving``, ``logspace``, or ``custom``."""

    def __init__(
        self,
        binning: Binning,
        method: BinMethod | str = BinMethod.linear,
    ) -> None:
        if not isinstance(binning, Binning):
            raise TypeError(f"'binning' must be of type '{type(binning)}'")
        object.__setattr__(self, "binning", binning)
        object.__setattr__(self, "method", BinMethod(method))

    @property
    def edges(self) -> NDArray:
        """Array of redshift bin edges."""
        return self.binning.edges

    @property
    def zmin(self) -> float:
        """Lowest redshift bin edge."""
        return float(self.binning.edges[0])

    @property
    def zmax(self) -> float:
        """Highest redshift bin edge."""
        return float(self.binning.edges[-1])

    @property
    def num_bins(self) -> int:
        """Number of redshift bins."""
        return len(self.binning)

    @property
    def closed(self) -> Closed:
        """String indicating if the bin edges are closed on the ``left`` or the
        ``right`` side."""
        return self.binning.closed

    @property
    def is_custom(self) -> bool:
        """Whether the bin edges are provided by the user."""
        return self.method == "custom"

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, Any], cosmology: TypeCosmology | None = None
    ) -> BinningConfig:
        """
        Restore the class instance from a python dictionary.

        Args:
            the_dict:
                Dictionary containing all required data attributes to restore
                the instance, see also :meth:`to_dict()`.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            Restored class instance.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        if the_dict["method"] == "custom":
            edges = the_dict.pop("edges")
            closed = the_dict.pop("closed")
            binning = Binning(edges, closed=closed)
            return cls(binning, **the_dict)

        return cls.create(**the_dict, cosmology=cosmology)

    def to_dict(self) -> dict[str, Any]:
        if self.is_custom:
            the_dict = dict(edges=self.binning.edges)

        else:
            the_dict = dict(
                zmin=self.zmin,
                zmax=self.zmax,
                num_bins=self.num_bins,
            )

        the_dict["method"] = str(self.method)
        the_dict["closed"] = str(self.closed)
        return the_dict

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.method == other.method and self.binning == other.binning

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = [
            Parameter(
                name="zmin",
                help="Lowest redshift bin edge to generate.",
                type=float,
            ),
            Parameter(
                name="zmax",
                help="Highest redshift bin edge to generate.",
                type=float,
            ),
            Parameter(
                name="num_bins",
                help="Number of redshift bins to generate.",
                type=int,
                default=30,
            ),
            Parameter(
                name="method",
                help="Method used to generate the bin edges, must be either of ``linear``, ``comoving``, ``logspace``, or ``custom``.",
                type=str,
                choices=get_options(Closed),
                default=str(BinMethod.linear),
            ),
            Parameter(
                name="edges",
                help="Use these custom bin edges instead of generating them.",
                type=float,
                is_sequence=True,
                default=None,
            ),
            Parameter(
                name="closed",
                help="String indicating if the bin edges are closed on the ``left`` or the ``right`` side.",
                type=str,
                choices=get_options(Closed),
                default=str(Closed.right),
            ),
        ]
        return ParamSpec(params)

    @classmethod
    def create(
        cls,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
        num_bins: int = 30,
        method: BinMethod | str = BinMethod.linear,
        edges: Iterable[float] | None = None,
        closed: Closed | str = Closed.right,
        cosmology: TypeCosmology | None = None,
    ) -> BinningConfig:
        """
        Create a new instance with the given parameters.

        Keyword Args:
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
                Optional, cosmological model to use for distance computations.

        Returns:
            New configuration instance.

        .. note::
            All function parameters are optional, but either ``zmin`` and
            ``zmax`` (generate bin edges), or ``edges`` (custom bin edges) must
            be provided.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        auto_args_set = (zmin is not None, zmax is not None)
        custom_args_set = (edges is not None,)
        if not all(custom_args_set) and not all(auto_args_set):
            raise ConfigError("either 'edges' or 'zmin' and 'zmax' are required")

        closed = Closed(closed)

        if all(auto_args_set):  # generate bin edges
            if all(custom_args_set):
                warnings.warn(
                    "'zbins' set but ignored since 'zmin' and 'zmax' are provided"
                )
            method = BinMethod(method)
            bin_func = RedshiftBinningFactory(cosmology).get_method(method)
            binning = bin_func(zmin, zmax, num_bins, closed=closed)

        else:  # use provided bin edges
            method = BinMethod.custom
            binning = Binning(edges, closed=closed)

        return cls(binning, method=method)

    def modify(
        self,
        *,
        zmin: float | NotSet = NotSet,
        zmax: float | NotSet = NotSet,
        num_bins: int | NotSet = NotSet,
        method: BinMethod | str | NotSet = NotSet,
        edges: Iterable[float] | NotSet = NotSet,
        closed: Closed | str | NotSet = NotSet,
        cosmology: TypeCosmology | None | NotSet = NotSet,
    ) -> BinningConfig:
        """
        Create a new configuration instance with updated parameter values.

        Parameter values are only updated if they are provided as inputs to this
        function, otherwise they are retained from the original instance.

        Keyword Args:
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
                Optional, cosmological model to use for distance computations.

        Returns:
            New instance with updated redshift bins.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        if edges is NotSet:
            if method == "custom":
                raise ConfigError("'method' is 'custom' but no bin edges provided")
            the_dict = dict()
            the_dict["zmin"] = self.zmin if zmin is NotSet else zmin
            the_dict["zmax"] = self.zmax if zmax is NotSet else zmax
            the_dict["num_bins"] = self.num_bins if num_bins is NotSet else num_bins
            the_dict["method"] = self.method if method is NotSet else BinMethod(method)

        else:
            the_dict = dict(edges=edges)
            the_dict["method"] = BinMethod.custom

        the_dict["method"] = str(the_dict["method"])
        the_dict["closed"] = str(self.closed if closed is NotSet else Closed(closed))

        return type(self).from_dict(the_dict, cosmology=cosmology)
