"""
Implements a class that stores the configuration redshift bins used to split the
catalogs when measuring angular correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from yaw.binning import Binning
from yaw.config.base import ConfigError, Parameter, YawConfig
from yaw.cosmology import RedshiftBinningFactory, TypeCosmology
from yaw.options import BinMethod, Closed, NotSet

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import NDArray

__all__ = [
    "BinningConfig",
]


@dataclass(frozen=True)
class BinningConfig(YawConfig):
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

    _paramspec = (
        Parameter(
            name="zmin",
            help="Lowest redshift bin edge to generate (alternatively use 'edges').",
            type=float,
            default=None,
            nullable=True,
        ),
        Parameter(
            name="zmax",
            help="Highest redshift bin edge to generate (alternatively use 'edges').",
            type=float,
            default=None,
            nullable=True,
        ),
        Parameter(
            name="num_bins",
            help="Number of redshift bins to generate.",
            type=int,
            default=30,
        ),
        Parameter(
            name="method",
            help="Method used to generate the bin edges.",
            type=str,
            choices=BinMethod,
            default=BinMethod.linear,
        ),
        Parameter(
            name="edges",
            help="Use these custom bin edges instead of generating them.",
            type=float,
            is_sequence=True,
            default=None,
            nullable=True,
        ),
        Parameter(
            name="closed",
            help="String indicating the side of the bin intervals that are closed.",
            type=str,
            choices=Closed,
            default=Closed.right,
        ),
    )

    binning: Binning
    """Container for the redshift bins."""
    method: BinMethod
    """Method used to generate the bin edges, see :obj:`~yaw.options.BinMethod`
    for valid options."""

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
        """Indicating which side of the bin edges is a closed interval, see
        :obj:`~yaw.options.Closed` for valid options."""
        return self.binning.closed

    @property
    def is_custom(self) -> bool:
        """Whether the bin edges are provided by the user."""
        return self.method == "custom"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.method == other.method and self.binning == other.binning

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
        if not ("zmin" in the_dict and "zmax" in the_dict) and "edges" not in the_dict:
            raise ConfigError("either 'edges' or 'zmin' and 'zmax' are required")
        cls._check_dict_keys(the_dict)
        parsed = cls._parse_params(the_dict)

        if parsed["edges"] is not None:
            method = BinMethod.custom
            binning = Binning(parsed["edges"], closed=parsed["closed"])

        else:
            method = parsed["method"]
            binning = RedshiftBinningFactory(cosmology).get_method(method)(
                parsed["zmin"],
                parsed["zmax"],
                parsed["num_bins"],
                closed=parsed["closed"],
            )

        return cls(binning=binning, method=method)

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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
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
        the_dict = dict(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
        )
        return cls.from_dict(the_dict, cosmology=cosmology)

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
                Method used to generate the bin edges, see
                :obj:`~yaw.options.BinMethod` for valid options.
            edges:
                Use these custom bin edges instead of generating them.
            closed:
                Indicating which side of the bin edges is a closed interval, see
                :obj:`~yaw.options.Closed` for valid options.
            cosmology:
                Optional, cosmological model to use for distance computations.

        Returns:
            New instance with updated redshift bins.

        .. caution::
            This cosmology object is not stored with this instance, but should
            be managed by the top level :obj:`~yaw.Configuration` class.
        """
        the_dict = self.to_dict()
        updates = dict(
            zmin=zmin,
            zmax=zmax,
            num_bins=num_bins,
            method=method,
            edges=edges,
            closed=closed,
        )
        the_dict.update(kv for kv in updates.items() if kv[1] is not NotSet)
        return self.from_dict(the_dict, cosmology=cosmology)
