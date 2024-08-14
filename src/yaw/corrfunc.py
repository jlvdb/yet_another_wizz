from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Type, TypeVar

import h5py
import numpy as np
import pandas as pd
from deprecated import deprecated

from yaw.config import OPTIONS, ResamplingConfig

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from numpy.typing import NDArray
    from pandas import IntervalIndex

    from yaw.config import Configuration


class EstimatorError(Exception):
    pass


class Cts(ABC):
    """Base class to symbolically represent pair counts."""

    @property
    @abstractmethod
    def _hash(self) -> int:
        """Used for comparison.

        Equivalent pair counts types should have the same hash value, see
        :obj:`CtsMix`."""
        pass

    @property
    @abstractmethod
    def _str(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Cts):
            var1 = set(self._str.split("_"))
            var2 = set(other._str.split("_"))
            return not var1.isdisjoint(var2)
        return NotImplemented


class CtsDD(Cts):
    """Symbolic representation of data-data paircounts."""

    _str = "dd"
    _hash = 1


class CtsDR(Cts):
    """Symbolic representation of data-random paircounts."""

    _str = "dr"
    _hash = 2


class CtsRD(Cts):
    """Symbolic representation of random-data paircounts."""

    _str = "rd"
    _hash = 2


class CtsMix(Cts):
    """Symbolic representation of either data-random or random-data paircounts.

    >>> CtsDR() == CtsMix()
    True
    >>> CtsRD() == CtsMix()
    True
    """

    _str = "dr_rd"
    _hash = 2


class CtsRR(Cts):
    """Symbolic representation of random-random paircounts."""

    _str = "rr"
    _hash = 3


def cts_from_code(code: str) -> Cts:
    """Instantiate the correct :obj:`Cts` subclass from a string.

    Valid options are ``dd``, ``dr``, ``rd``, ``rr``, e.g.:

    >>> cts_from_code("dr")
    <CtsDR>
    """
    codes = dict(dd=CtsDD, dr=CtsDR, rd=CtsRD, rr=CtsRR)
    if code not in codes:
        raise ValueError(f"unknown pair counts '{code}'")
    return codes[code]()


class CorrelationEstimator(ABC):
    name: str
    """Full name of the estimator."""
    short: str
    """Get a short form representation of the estimator name."""
    requires: list[Cts]
    """Get a symbolic list of pair counts required to evaluate the estimator.
    """
    optional: list[Cts]
    """Get a symbolic list of optional pair counts that may be used to evaluate
    the estimator.
    """

    variants: list[CorrelationEstimator] = []
    """List of all implemented correlation estimators classes."""

    def __init_subclass__(cls, **kwargs):
        """Add all subclusses to the :obj:`variants` attribute list."""
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    @classmethod
    def _warn_enum_zero(cls, counts: NDArray):
        """Raise a warning if any value in the expression enumerator is zero"""
        if np.any(counts == 0.0):
            warnings.warn(f"estimator {cls.short} encontered zeros in enumerator")

    @classmethod
    @abstractmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Method that implements the estimator.

        Should call :meth:`_warn_enum_zero` to raise warnings on zero-division.
        """
        pass


class PeeblesHauser(CorrelationEstimator):
    """Implementation of the Peebles-Hauser correlation estimator
    :math:`\\frac{DD}{RR} - 1`.
    """

    name = "PeeblesHauser"
    short = "PH"
    requires = [CtsDD(), CtsRR()]
    optional = []

    @classmethod
    def eval(cls, *, dd: NDArray, rr: NDArray, **kwargs) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        cls._warn_enum_zero(rr)
        return dd / rr - 1.0


class DavisPeebles(CorrelationEstimator):
    """Implementation of the Davis-Peebles correlation estimator
    :math:`\\frac{DD}{DR} - 1`.

    .. Note::
        Accepts both :math:`DR` and :math:`RD` for the denominator.
    """

    name = "DavisPeebles"
    short = "DP"
    requires = [CtsDD(), CtsMix()]
    optional = []

    @classmethod
    def eval(cls, *, dd: NDArray, dr_rd: NDArray, **kwargs) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr_rd (:obj:`NDArray`):
                Either data-random or random-data pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        cls._warn_enum_zero(dr_rd)
        return dd / dr_rd - 1.0


class Hamilton(CorrelationEstimator):
    """Implementation of the Hamilton correlation estimator
    :math:`\\frac{DD \\times RR}{DR \\times RD} - 1`.
    """

    name = "Hamilton"
    short = "HM"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    @classmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr (:obj:`NDArray`):
                Data-random pair counts (normalised).
            rd (:obj:`NDArray`, optional):
                Random-data pair counts (normalised). If not provided, use
                ``dr`` instead.
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        rd = dr if rd is None else rd
        enum = dr * rd
        cls._warn_enum_zero(enum)
        return (dd * rr) / enum - 1.0


class LandySzalay(CorrelationEstimator):
    """Implementation of the Landy-Szalay correlation estimator
    :math:`\\frac{DD - (DR + RD)}{RR} + 1`.
    """

    name = "LandySzalay"
    short = "LS"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    @classmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr (:obj:`NDArray`):
                Data-random pair counts (normalised).
            rd (:obj:`NDArray`, optional):
                Random-data pair counts (normalised). If not provided, use
                ``dr`` instead.
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        rd = dr if rd is None else rd
        cls._warn_enum_zero(rr)
        return (dd - (dr + rd) + rr) / rr


@dataclass(frozen=True)
class CorrFunc(PatchedQuantity, BinnedQuantity, HDFSerializable):
    """Container object for measured correlation pair counts.

    Container returned by :meth:`~yaw.old_catalogs.BaseCatalog.correlate` that
    computes the correlations between data catalogs. The correlation function
    can be computed from four kinds of pair counts, data-data (DD), data-random
    (DR), random-data (RD), and random-random (RR).

    .. Note::
        DD is always required, but DR, RD, and RR are optional as long as at
        least one is provided.

    Provides methods to read and write data to disk and compute the actual
    correlation function values (see :class:`~yaw.CorrData`) using spatial
    resampling (see :class:`~yaw.ResamplingConfig`).

    The container supports comparison with ``==`` and ``!=`` on the pair count
    level. The supported arithmetic operations between two correlation
    functions, addition and subtraction, are applied between all internally
    stored pair counts data. The same applies to rescaling of the counts by a
    scalar, see some examples below.

    .. rubric:: Examples

    Create a new instance by sampling a correlation function:

    >>> from yaw.examples import w_sp
    >>> dd, dr = w_sp.dd, w_sp.dr  # get example data-data and data-rand counts
    >>> corr = yaw.CorrFunc(dd=dd, dr=dr)
    >>> corr
    CorrFunc(n_bins=30, z='0.070...1.420', dd=True, dr=True, rd=False, rr=False, n_patches=64)

    Access the pair counts:

    >>> corr.dd
    NormalisedCounts(n_bins=30, z='0.070...1.420', n_patches=64)

    Check if it is an autocorrelation function measurement:

    >>> corr.auto
    False

    Check which pair counts are available to compute the correlation function:

    >>> corr.estimators
    {'DP': yaw.correlation.estimators.DavisPeebles}

    Sample the correlation function

    >>> corr.sample()  # uses the default ResamplingConfig
    CorrData(n_bins=30, z='0.070...1.420', n_samples=64, method='jackknife')

    Note how the indicated shape changes when a patch subset is selected:

    >>> corr.patches[:10]
    CorrFunc(n_bins=30, z='0.070...1.420', dd=True, dr=True, rd=False, rr=False, n_patches=10)

    Note how the indicated redshift range and shape change when a bin subset is
    selected:

    >>> corr.bins[:3]
    CorrFunc(n_bins=3, z='0.070...0.205', dd=True, dr=True, rd=False, rr=False, n_patches=64)

    Args:
        dd (:obj:`~yaw.correlation.paircounts.NormalisedCounts`):
            Pair counts from a data-data count measurement.
        dr (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a data-random count measurement.
        rd (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a random-data count measurement.
        rr (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a random-random count measurement.
    """

    dd: NormalisedCounts
    """Pair counts for a data-data correlation measurement"""
    dr: NormalisedCounts | None = field(default=None)
    """Pair counts from a data-random count measurement."""
    rd: NormalisedCounts | None = field(default=None)
    """Pair counts from a random-data count measurement."""
    rr: NormalisedCounts | None = field(default=None)
    """Pair counts from a random-random count measurement."""

    def __post_init__(self) -> None:
        # check if any random pairs are required
        if self.dr is None and self.rd is None and self.rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: NormalisedCounts | None = getattr(self, kind)
            if pairs is None:
                continue
            try:
                self.dd.is_compatible(pairs, require=True)
                assert self.dd.n_patches == pairs.n_patches
            except (ValueError, AssertionError) as e:
                raise ValueError(
                    f"pair counts '{kind}' and 'dd' are not compatible"
                ) from e

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        pairs = f"dd=True, dr={self.dr is not None}, "
        pairs += f"rd={self.rd is not None}, rr={self.rr is not None}"
        other = f"n_patches={self.n_patches}"
        return f"{string}, {pairs}, {other})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            for cfield in fields(self):
                kind = cfield.name
                if getattr(self, kind) != getattr(other, kind):
                    return False
            return True
        return NotImplemented

    def __add__(self, other: object) -> CorrFunc:
        if isinstance(other, self.__class__):
            # check that the pair counts are set consistently
            kinds = []
            for cfield in fields(self):
                kind = cfield.name
                self_set = getattr(self, kind) is not None
                other_set = getattr(other, kind) is not None
                if (self_set and not other_set) or (not self_set and other_set):
                    raise ValueError(
                        f"pair counts for '{kind}' not set for both operands"
                    )
                elif self_set and other_set:
                    kinds.append(kind)

            kwargs = {
                kind: getattr(self, kind) + getattr(other, kind) for kind in kinds
            }
            return self.__class__(**kwargs)
        return NotImplemented

    def __radd__(self, other: object) -> CorrFunc:
        if np.isscalar(other) and other == 0:
            return self
        return other.__add__(self)

    def __mul__(self, other: object) -> CorrFunc:
        if np.isscalar(other) and not isinstance(other, (bool, np.bool_)):
            # check that the pair counts are set consistently
            kwargs = {}
            for cfield in fields(self):
                kind = cfield.name
                counts = getattr(self, kind)
                if counts is not None:
                    kwargs[kind] = counts * other
            return self.__class__(**kwargs)
        return NotImplemented

    @property
    def auto(self) -> bool:
        """Whether the stored data are from an autocorrelation measurement."""
        return self.dd.auto

    @property
    def bins(self) -> Indexer[TypeIndex, CorrFunc]:
        def builder(inst: CorrFunc, item: TypeIndex) -> CorrFunc:
            if isinstance(item, int):
                item = [item]
            kwargs = {}
            for cfield in fields(inst):
                pairs: NormalisedCounts | None = getattr(inst, cfield.name)
                if pairs is None:
                    kwargs[cfield.name] = None
                else:
                    kwargs[cfield.name] = pairs.bins[item]
            return CorrFunc(**kwargs)

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, CorrFunc]:
        def builder(inst: CorrFunc, item: TypeIndex) -> CorrFunc:
            kwargs = {}
            for cfield in fields(inst):
                counts: NormalisedCounts | None = getattr(inst, cfield.name)
                if counts is not None:
                    counts = counts.patches[item]
                kwargs[cfield.name] = counts
            return CorrFunc(**kwargs)

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self.dd.get_binning()

    @property
    def n_patches(self) -> int:
        return self.dd.n_patches

    def is_compatible(self, other: CorrFunc, require: bool = False) -> bool:
        """Check whether this instance is compatible with another instance.

        Ensures that the redshift binning and the number of patches are
        identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (:obj:`bool`)
                Raise a ValueError if any of the checks fail.

        Returns:
            :obj:`bool`
        """
        if self.dd.n_patches != other.dd.n_patches:
            if require:
                raise ValueError("number of patches does not agree")
            return False
        return self.dd.is_compatible(other.dd, require)

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        """Get a listing of correlation estimators implemented, depending on
        which pair counts are available.

        Returns:
            :obj:`dict`: Mapping from correlation estimator name abbreviation to
            correlation function class.
        """
        # figure out which of dd, dr, ... are not None
        available = set()
        # iterate all dataclass attributes that are in __init__
        for attr in fields(self):
            if getattr(self, attr.name) is not None:
                available.add(cts_from_code(attr.name))
        # check which estimators are supported
        estimators = {}
        for estimator in CorrelationEstimator.variants:  # registered estimators
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self, estimator: str | None = None
    ) -> type[CorrelationEstimator]:
        options = self.estimators
        if estimator is None:
            for shortname in ["LS", "DP", "PH"]:  # preferred hierarchy
                if shortname in options:
                    estimator = shortname
                    break
        estimator = estimator.upper()
        if estimator not in options:
            try:
                index = [e.short for e in CorrelationEstimator.variants].index(
                    estimator
                )
                est_class = CorrelationEstimator.variants[index]
            except ValueError as e:
                raise ValueError(f"invalid estimator '{estimator}'") from e
            # determine which pair counts are missing
            for attr in fields(self):
                name = attr.name
                cts = cts_from_code(name)
                if getattr(self, name) is None and cts in est_class.requires:
                    raise EstimatorError(f"estimator requires {name}")
        # select the correct estimator
        cls = options[estimator]
        logger.debug(
            "selecting estimator '%s' from %s", cls.short, "/".join(self.estimators)
        )
        return cls

    def _getattr_from_cts(self, cts: Cts) -> NormalisedCounts | None:
        if isinstance(cts, CtsMix):
            for code in str(cts).split("_"):
                value = getattr(self, code)
                if value is not None:
                    break
            return value
        else:
            return getattr(self, str(cts))

    @deprecated(reason="renamed to CorrFunc.sample", version="2.3.1")
    def get(self, *args, **kwargs):
        """
        .. deprecated:: 2.3.1
            Renamed to :meth:`sample`.
        """
        return self.sample(*args, **kwargs)  # pragma: no cover

    def sample(
        self,
        config: ResamplingConfig | None = None,
        *,
        estimator: str | None = None,
        info: str | None = None,
    ) -> CorrData:
        """Compute the correlation function from the stored pair counts,
        including an error estimate from spatial resampling of patches.

        Args:
            config (:obj:`~yaw.ResamplingConfig`):
                Specify the resampling method and its configuration.

        Keyword Args:
            estimator (:obj:`str`, optional):
                The name abbreviation for the correlation estimator to use.
                Defaults to Landy-Szalay if RR is available, otherwise to
                Davis-Peebles.
            info (:obj:`str`, optional):
                Descriptive text passed on to the output :obj:`CorrData`
                object.

        Returns:
            :obj:`CorrData`:
                Correlation function data, including redshift binning, function
                values and samples.
        """
        if config is None:
            config = ResamplingConfig()
        est_fun = self._check_and_select_estimator(estimator)
        logger.debug("computing correlation and %s samples", config.method)
        # get the pair counts for the required terms (DD, maybe DR and/or RR)
        required_data = {}
        required_samples = {}
        for cts in est_fun.requires:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                required_data[str(cts)] = pairs.data
                required_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # get the pair counts for the optional terms (e.g. RD)
        optional_data = {}
        optional_samples = {}
        for cts in est_fun.optional:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                optional_data[str(cts)] = pairs.data
                optional_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # evaluate the correlation estimator
        data = est_fun.eval(**required_data, **optional_data)
        samples = est_fun.eval(**required_samples, **optional_samples)
        return CorrData(
            binning=self.get_binning(),
            data=data,
            samples=samples,
            method=config.method,
            info=info,
        )

    @classmethod
    def from_hdf(cls, source: h5py.File | h5py.Group) -> CorrFunc:
        def _try_load(root: h5py.Group, name: str) -> NormalisedCounts | None:
            try:
                return NormalisedCounts.from_hdf(root[name])
            except KeyError:
                return None

        dd = NormalisedCounts.from_hdf(source["data_data"])
        dr = _try_load(source, "data_random")
        rd = _try_load(source, "random_data")
        rr = _try_load(source, "random_random")
        return cls(dd=dd, dr=dr, rd=rd, rr=rr)

    def to_hdf(self, dest: h5py.File | h5py.Group) -> None:
        group = dest.create_group("data_data")
        self.dd.to_hdf(group)
        group_names = dict(dr="data_random", rd="random_data", rr="random_random")
        for kind, name in group_names.items():
            data: NormalisedCounts | None = getattr(self, kind)
            if data is not None:
                group = dest.create_group(name)
                data.to_hdf(group)
        dest.create_dataset("n_patches", data=self.n_patches)

    @classmethod
    def from_file(cls, path: TypePathStr) -> CorrFunc:
        logger.debug("reading pair counts from '%s'", path)
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        logger.info("writing pair counts to '%s'", path)
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)

    def concatenate_patches(self, *cfs: CorrFunc) -> CorrFunc:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_patches(*other_pcounts)
        return self.__class__(**merged)

    def concatenate_bins(self, *cfs: CorrFunc) -> CorrFunc:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_bins(*other_pcounts)
        return self.__class__(**merged)
