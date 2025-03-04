"""
Implements the low-level containers for pair counts.

Pair counts are stored for pairs of spatial patches and per redshift bin,
forming a 3-dim array of measurements, stored as NormalisedCounts. Internally,
the normalised counts store the actual sum of pair weights and the number of
objects stored in each patch, from which the later normalisation of the pair
counts is computed.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from yaw.binning import Binning, load_legacy_binning
from yaw.correlation.corrdata import SampledData
from yaw.utils import (
    HDF_COMPRESSION,
    is_legacy_dataset,
    load_version_tag,
    write_version_tag,
)
from yaw.utils.abc import BinwiseData, HdfSerializable, PatchwiseData
from yaw.utils.parallel import Broadcastable

if TYPE_CHECKING:
    from typing import Any

    from h5py import Group
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.utils.abc import TypeSliceIndex

__all__ = [
    "PatchedSumWeights",
    "PatchedCounts",
    "NormalisedCounts",
    "NormalisedScalarCounts",
]


class BinwisePatchwiseArray(BinwiseData, PatchwiseData, HdfSerializable, Broadcastable):
    """Meta-class for correlation function pair counts, recorded in bins of
    redshift and for pairs of patches."""

    __slots__ = ()

    @property
    @abstractmethod
    def auto(self) -> bool:
        """Whether the pair counts originate from an autocorrelation
        measurement."""
        pass

    def __repr__(self) -> str:
        items = (
            f"auto={self.auto}",
            f"binning={self.binning}",
            f"num_patches={self.num_patches}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        """
        Checks if two containers have the same redshift binning and number of
        spatial patches.

        Args:
            other:
                Another instance of this class to compare to, returns ``False``
                if instance types do not match.

        Keyword Args:
            require:
                Whether to raise exceptions if any of the checks fail.

        Returns:
            Whether the number of patches is identical if ``require=False``.

        Raises:
            TypeError:
                If ``require=True`` and type of ``other`` does match this class.
            ValueError:
                If ``require=True`` and binning and the number of patches is not
                identical.
        """
        binnings_compatible = BinwiseData.is_compatible(self, other, require=require)
        patches_compatible = PatchwiseData.is_compatible(self, other, require=require)
        return binnings_compatible and patches_compatible

    @abstractmethod
    def get_array(self) -> NDArray:
        """
        Represent the internal data as numpy array with shape
        (:obj:`num_bins`, :obj:`num_patches`, :obj:`num_patches`).

        I.e. the first array element contains the data associated with the first
        redshift bin and pairing the first patch with itself.

        Returns:
            Internal data represented as numpy array.
        """
        pass

    def sample_patch_sum(self) -> SampledData:
        """
        Compute the sum over all patches and leave-one-out jackknife samples.

        I.e. marginalise over the patch axes and return a 1-dim array with
        length :obj:`num_bins`.

        Returns:
            Sum over patches and jackknife samples thereof packed in an
            instance of :obj:`~yaw.correlation.corrdata.SampledData`.
        """
        bin_patch_array = self.get_array()

        # sum over all (total of num_paches**2) pairs of patches per bin
        sum_patches = np.einsum("bij->b", bin_patch_array)

        # jackknife efficiency trick: take the sum from above and, for the i-th
        # sample, subtract the contribution of pairs containing the i-th patch:
        # 1) repeat the sum to final shape of samples (num_samples, num_bins)
        sum_tiled = np.tile(sum_patches, (self.num_patches, 1))
        # 2) compute the sum over pairs formed by the i-th patch with all others
        row_sum = np.einsum("bij->jb", bin_patch_array)
        # 3) compute the sum over pairs formed by all other patches with i-th
        col_sum = np.einsum("bij->ib", bin_patch_array)
        # 4) compute the sum over diagonals because it is counted twice in 2 & 3
        diag = np.einsum("bii->ib", bin_patch_array)
        samples = sum_tiled - row_sum - col_sum + diag

        return SampledData(self.binning, sum_patches, samples)


class PatchedSumWeights(BinwisePatchwiseArray):
    """
    Stores the sum of weights in spatial patches from catalogs in a correlation
    measurement.

    The sum of weights are stored separately for the first and second catalog,
    each per patch and per redshift bin. The product of these numbers is used
    to normalised correlation pair counts between patches. This normalisation
    factor in bins or redshifts, including jackknife samples thereof, can be
    obtained by calling :meth:`sample_patch_sum`, which sums over all possible
    pairs of patches from the first and second catalog.

    Implements comparison with the ``==`` operator.

    Args:
        binning:
            Redshift bins used when counting pairs between patches.
        sum_weights1:
            Sum of weights in patches of catalog 1, array with shape
            (:obj:`num_bins`, :obj:`num_patches`).
        sum_weights2:
            Sum of weights in patches of catalog 2, array with shape
            (:obj:`num_bins`, :obj:`num_patches`).

    Keyword Args:
        auto:
            Whether the pair counts originate from an autocorrelation
            measurement.
    """

    __slots__ = ("binning", "auto", "sum_weights1", "sum_weights2")

    binning: Binning
    """Accessor for the redshift :obj:`~yaw.Binning` attribute."""
    sum_weights1: NDArray
    """Sum of weights in patches of catalog 1, array with shape
    (:obj:`num_bins`, :obj:`num_patches`)."""
    sum_weights2: NDArray
    """Sum of weights in patches of catalog 2, array with shape
    (:obj:`num_bins`, :obj:`num_patches`)."""
    auto: bool
    """Whether the pair counts originate from an autocorrelation measurement."""

    def __init__(
        self,
        binning: Binning,
        sum_weights1: NDArray,
        sum_weights2: NDArray,
        *,
        auto: bool,
    ) -> None:
        self.binning = binning
        self.auto = auto

        if sum_weights1.ndim != sum_weights2.ndim != 2:
            raise ValueError("'sum_weights1/2' must be two-dimensional")
        if sum_weights1.shape != sum_weights2.shape:
            raise ValueError(
                "'sum_weights1' and 'sum_weights2' must have the same shape"
            )
        if sum_weights1.shape[0] != self.num_bins:
            raise ValueError("first dimension of 'sum_weights1/2' must match 'binning'")

        self.sum_weights1 = sum_weights1.astype(np.float64)
        self.sum_weights2 = sum_weights2.astype(np.float64)

    @classmethod
    def from_hdf(cls: type[Self], source: Group) -> Self:
        new = cls.__new__(cls)
        new.auto = source["auto"][()]

        if is_legacy_dataset(source):
            new.sum_weights1 = np.transpose(source["totals1"])
            new.sum_weights2 = np.transpose(source["totals2"])
            new.binning = load_legacy_binning(source)
        else:
            new.sum_weights1 = source["sum_weights1"][:]
            new.sum_weights2 = source["sum_weights2"][:]
            new.binning = Binning.from_hdf(source["binning"])

        return new

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        self.binning.to_hdf(dest.create_group("binning"))
        dest.create_dataset("auto", data=self.auto)

        dest.create_dataset("sum_weights1", data=self.sum_weights1, **HDF_COMPRESSION)
        dest.create_dataset("sum_weights2", data=self.sum_weights2, **HDF_COMPRESSION)

    @property
    def num_patches(self) -> int:
        return self.sum_weights1.shape[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.sum_weights1, other.sum_weights1)
            and np.array_equal(self.sum_weights2, other.sum_weights2)
            and self.auto == other.auto
        )

    def _make_bin_slice(self, item: TypeSliceIndex) -> Self:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]
        return type(self)(
            binning, self.sum_weights1[item], self.sum_weights2[item], auto=self.auto
        )

    def _make_patch_slice(self, item: TypeSliceIndex) -> Self:
        if isinstance(item, int):
            item = [item]
        return type(self)(
            self.binning,
            self.sum_weights1[:, item],
            self.sum_weights2[:, item],
            auto=self.auto,
        )

    def get_array(self) -> NDArray:
        """
        Represent the internal data as numpy array with shape
        (:obj:`num_bins`, :obj:`num_patches`, :obj:`num_patches`).

        Construct array of product of sum of weights for patch pairs, i.e. an
        array that holds the product of the sum from patch :math:`i` from
        catalog 1 and patch :math:`j` from catalog 2 in the :math:`b`-th
        redshift bin at index ``[b, i, j]``.

        Returns:
            Internal data represented as numpy array.
        """
        array = np.einsum("bi,bj->bij", self.sum_weights1, self.sum_weights2)

        if self.auto:
            # for auto-correlation totals we need to set lower trinagle to zero
            array = np.triu(array)
            # additionally we must halve the totals on diagonals for every bin
            np.einsum("bii->bi", array)[:] *= 0.5  # view of original array

        return array


class PatchedCounts(BinwisePatchwiseArray):
    """
    Stores the pair counts in spatial patches from catalogs in a correlation
    measurement.

    The pair counts are stored per redshift bin and combination of patches. The
    total counts per redshift bin, including jackknife samples thereof, can
    be obtained by calling :meth:`sample_patch_sum`, which sums over all
    possible pairs of patches from the first and second catalog.

    Implements comparison with the ``==`` operator, addition of counts with the
    ``+``/``+=`` operator and scalar multiplication of the counts with the ``*``
    operator.

    Args:
        binning:
            Redshift bins used when counting pairs between patches.
        counts:
            Array of with pair counts in bins of redshift between combinations
            of patch pairs from both catalos, numpy array with shape
            (:obj:`num_bins`, :obj:`num_patches`, :obj:`num_patches`).

    Keyword Args:
        auto:
            Whether this instance is intended for an autocorrelation
            measurement.
    """

    __slots__ = ("binning", "counts", "auto")

    binning: Binning
    """Accessor for the redshift :obj:`~yaw.Binning` attribute."""
    counts: NDArray
    """Pair counts between patches of catalog 1 and 2 per redshift bin, array
    with shape (:obj:`num_bins`, :obj:`num_patches`, :obj:`num_patches`)."""
    auto: bool
    """Whether the pair counts originate from an autocorrelation measurement."""

    def __init__(self, binning: Binning, counts: NDArray, *, auto: bool) -> None:
        self.binning = binning
        self.auto = auto

        if counts.ndim != 3:
            raise ValueError("'counts' must be three-dimensional")
        if counts.shape[0] != self.num_bins:
            raise ValueError("first dimension of 'counts' must match 'binning'")
        if counts.shape[1] != counts.shape[2]:
            raise ValueError(
                "'counts' must have shape (num_bins, num_patches, num_patches)"
            )

        self.counts = counts.astype(np.float64)

    @classmethod
    def zeros(
        cls: type[Self], binning: Binning, num_patches: int, *, auto: bool
    ) -> Self:
        """
        Create a new instance with all pair counts initialised to zero.

        Args:
            binning:
                Redshift bins used when counting pairs between patches.
            num_patches:
                The number of patches in the input catalogs used for the
                correlation measurement.

        Keyword Args:
            auto:
                Whether this instance is intended for an autocorrelation
                measurement.

        Returns:
            Initialised :obj:`PatchedCounts` instance.
        """
        num_bins = len(binning)
        counts = np.zeros((num_bins, num_patches, num_patches))
        return cls(binning, counts, auto=auto)

    @classmethod
    def from_hdf(cls: type[Self], source: Group) -> Self:
        auto = source["auto"][()]

        if is_legacy_dataset(source):
            binning = load_legacy_binning(source)

            num_patches = source["n_patches"][()]
            patch_pairs = source["keys"][:]
            binned_counts = source["data"][:]

        else:
            binning = Binning.from_hdf(source["binning"])

            num_patches = source["num_patches"][()]
            patch_pairs = source["patch_pairs"][:]
            binned_counts = source["binned_counts"][:]

        new = cls.zeros(binning, num_patches, auto=auto)
        for (patch_id1, patch_id2), counts in zip(patch_pairs, binned_counts):
            new.set_patch_pair(patch_id1, patch_id2, counts)

        return new

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)

        self.binning.to_hdf(dest.create_group("binning"))
        dest.create_dataset("auto", data=self.auto)
        dest.create_dataset("num_patches", data=self.num_patches)

        is_nonzero = np.any(self.counts, axis=0)
        patch_ids1, patch_ids2 = np.nonzero(is_nonzero)
        patch_pairs = np.column_stack([patch_ids1, patch_ids2])
        dest.create_dataset("patch_pairs", data=patch_pairs, **HDF_COMPRESSION)

        counts = self.counts[:, patch_ids1, patch_ids2]
        binned_counts = np.moveaxis(counts, 0, -1)  # match patch_pairs
        dest.create_dataset("binned_counts", data=binned_counts, **HDF_COMPRESSION)

    @property
    def num_patches(self) -> int:
        return self.counts.shape[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.counts, other.counts)
            and self.auto == other.auto
        )

    def _make_bin_slice(self, item: TypeSliceIndex) -> Self:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]

        return type(self)(binning, self.counts[item], auto=self.auto)

    def _make_patch_slice(self, item: TypeSliceIndex) -> Self:
        if isinstance(item, int):
            item = [item]

        return type(self)(self.binning, self.counts[:, item, item], auto=self.auto)

    def get_array(self) -> NDArray:
        return self.counts

    def set_patch_pair(
        self, patch_id1: int, patch_id2: int, counts_binned: NDArray
    ) -> None:
        """
        Set the correlation pair counts between two patches in each redshift
        bin.

        Args:
            patch_id1:
                ID/index of the patch from catalog 1.
            patch_id2:
                ID/index of the patch from catalog 2.
            counts_binned:
                Array with pair counts per redshift bin between patches with
                length :obj:`num_patches`.
        """
        self.counts[:, patch_id1, patch_id2] = counts_binned


class BaseNormalisedCounts(BinwisePatchwiseArray):
    """Base class for storing normalised pair counts."""

    __slots__ = ("_counts", "_weights")

    _counts: BinwisePatchwiseArray
    """Container storing correlation pair counts."""
    _weights: BinwisePatchwiseArray
    """Container defining normalisation for pair counts."""

    def _init(
        self, counts: BinwisePatchwiseArray, weights: BinwisePatchwiseArray
    ) -> None:
        if counts.num_patches != weights.num_patches:
            raise ValueError(
                "number of patches of counts- and weights-container does not match"
            )
        if counts.num_bins != weights.num_bins:
            raise ValueError(
                "number of bins of counts- and weights-container does not match"
            )

        self._counts = counts
        self._weights = weights

    @classmethod
    @abstractmethod
    def _get_hdf_names(cls, version_tag: str) -> tuple[str, str]:
        """Get the name of the HDF5 groups that store the counts and weights."""
        pass

    @classmethod
    @abstractmethod
    def from_hdf(cls: type[Self], source: Group) -> Self:
        pass

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        counts_name, weights_name = self._get_hdf_names(load_version_tag(dest))

        group = dest.create_group(counts_name)
        self._counts.to_hdf(group)

        group = dest.create_group(weights_name)
        self._weights.to_hdf(group)

    @property
    def binning(self) -> Binning:
        return self._counts.binning

    @property
    def auto(self) -> bool:
        return self._counts.auto

    @property
    def num_patches(self) -> int:
        return self._counts.num_patches

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if type(self) is not type(other):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self._counts.is_compatible(other._counts, require=require)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return self._counts == other._counts and self._weights == other._weights

    def _make_bin_slice(self, item: TypeSliceIndex) -> Self:
        _counts = self._counts.bins[item]
        _weights = self._weights.bins[item]
        return type(self)(_counts, _weights)

    def _make_patch_slice(self, item: TypeSliceIndex) -> Self:
        _counts = self._counts.patches[item]
        _weights = self._weights.patches[item]
        return type(self)(_counts, _weights)

    def get_array(self) -> NDArray:
        """
        Represent normalised pair counts as numpy array with shape
        (:obj:`num_bins`, :obj:`num_patches`, :obj:`num_patches`).

        I.e. the first array element contains the data associated with the first
        redshift bin and pairing the first patch with itself.

        .. note::
            The normalisation is computed from all patches and not per patch.

        Returns:
            Internal data represented as numpy array.
        """
        _counts = self._counts.get_array()
        _weights = self._weights.sample_patch_sum()
        return _counts / _weights.data[:, np.newaxis, np.newaxis]

    def sample_patch_sum(self) -> SampledData:
        _counts = self._counts.sample_patch_sum()
        _weights = self._weights.sample_patch_sum()

        data = _counts.data / _weights.data
        samples = _counts.samples / _weights.samples
        return SampledData(self.binning, data, samples)


class NormalisedCounts(BaseNormalisedCounts):
    """
    Stores normalised pair counts from a correlation measurement.

    This class stores the raw pair counts (:obj:`counts`) and the product of the
    sum of weights (:obj:`sum_weights`) per redshift bin and combination of
    patch pairs between the two catalogs use for the pair counting. The total
    of the normalised counts per redshift bin, including jackknife samples
    thereof, can be obtained by calling :meth:`sample_patch_sum`. The method
    computes the sum of pair counts over all possible pairs of patches and
    normalises them by dividing them by the product of the sum of weights from
    catalog 1 and 2.

    Implements comparison with the ``==`` operator.

    Args:
        counts:
            Container of correlation pair counts.
        sum_weights:
            Container of sum of weights in patches of catalogs 1 and 2.
    """

    __slots__ = ("_counts", "_weights")

    def __init__(self, counts: PatchedCounts, sum_weights: PatchedSumWeights) -> None:
        self._init(counts, sum_weights)

    @property
    def counts(self) -> PatchedCounts:
        """Container of correlation pair counts."""
        return self._counts

    @property
    def sum_weights(self) -> PatchedSumWeights:
        """Container of sum of weights in patches of catalogs 1 and 2."""
        return self._weights

    @classmethod
    def _get_hdf_names(cls, version_tag: str) -> tuple[str, str]:
        if version_tag.startswith("2"):
            return ("count", "total")
        return ("counts", "sum_weights")

    @classmethod
    def from_hdf(cls: type[Self], source: Group) -> Self:
        counts_name, weights_name = cls._get_hdf_names(load_version_tag(source))
        _counts = PatchedCounts.from_hdf(source[counts_name])
        _weights = PatchedSumWeights.from_hdf(source[weights_name])
        return cls(_counts, _weights)


class NormalisedScalarCounts(BaseNormalisedCounts):
    """
    Stores normalised pair counts from a correlation measurement including a
    scalar field.

    This class stores the pair counts weighted by the scalar field and the
    regular (number) pair counts, which are used as normalisation. The counts
    are recorded per redshift bin and combination of patch pairs between the two
    catalogs use for the pair counting. The total of the normalised counts per
    redshift bin, including jackknife samples thereof, can be obtained by
    calling :meth:`sample_patch_sum`.

    Implements comparison with the ``==`` operator.

    Args:
        kappa_counts:
            Container of correlation pair counts with scalar field weights.
        number_counts:
            Container of regular pair counts used as normalisation.
    """

    __slots__ = ("_counts", "_weights")

    def __init__(
        self, kappa_counts: PatchedCounts, number_counts: PatchedCounts
    ) -> None:
        self._init(kappa_counts, number_counts)

    @property
    def kappa_counts(self) -> PatchedCounts:
        """Container of correlation pair counts with scalar field weights."""
        return self._counts

    @property
    def number_counts(self) -> PatchedCounts:
        """Container of regular pair counts used as normalisation."""
        return self._weights

    @classmethod
    def _get_hdf_names(cls, version_tag: str) -> tuple[str, str]:
        return ("kappa_counts", "number_counts")

    @classmethod
    def from_hdf(cls: type[Self], source: Group) -> Self:
        counts_name, weights_name = cls._get_hdf_names(load_version_tag(source))
        _counts = PatchedCounts.from_hdf(source[counts_name])
        _weights = PatchedCounts.from_hdf(source[weights_name])
        return cls(_counts, _weights)
