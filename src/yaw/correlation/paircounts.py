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
from yaw.utils import HDF_COMPRESSION, is_legacy_dataset, write_version_tag
from yaw.utils.abc import BinwiseData, HdfSerializable, PatchwiseData
from yaw.utils.parallel import Broadcastable

if TYPE_CHECKING:
    from typing import Any

    from h5py import Group
    from numpy.typing import NDArray

    from yaw.utils.abc import TypeSliceIndex

__all__ = [
    "PatchedTotals",
    "PatchedCounts",
    "NormalisedCounts",
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


class PatchedTotals(BinwisePatchwiseArray):
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
        totals1:
            Sum of weights in patches of catalog 1, array with shape
            (:obj:`num_bins`, :obj:`num_patches`).
        totals2:
            Sum of weights in patches of catalog 2, array with shape
            (:obj:`num_bins`, :obj:`num_patches`).

    Keyword Args:
        auto:
            Whether the pair counts originate from an autocorrelation
            measurement.
    """

    __slots__ = ("binning", "auto", "totals1", "totals2")

    binning: Binning
    """Accessor for the redshift :obj:`~yaw.Binning` attribute."""
    totals1: NDArray
    """Sum of weights in patches of catalog 1, array with shape
    (:obj:`num_bins`, :obj:`num_patches`)."""
    totals2: NDArray
    """Sum of weights in patches of catalog 2, array with shape
    (:obj:`num_bins`, :obj:`num_patches`)."""
    auto: bool
    """Whether the pair counts originate from an autocorrelation measurement."""

    def __init__(
        self, binning: Binning, totals1: NDArray, totals2: NDArray, *, auto: bool
    ) -> None:
        self.binning = binning
        self.auto = auto

        if totals1.ndim != totals2.ndim != 2:
            raise ValueError("'totals1/2' must be two-dimensional")
        if totals1.shape != totals2.shape:
            raise ValueError("'totals1' and 'totals2' must have the same shape")
        if totals1.shape[0] != self.num_bins:
            raise ValueError("first dimension of 'totals1/2' must match 'binning'")

        self.totals1 = totals1.astype(np.float64)
        self.totals2 = totals2.astype(np.float64)

    @classmethod
    def from_hdf(cls, source: Group) -> PatchedTotals:
        new = cls.__new__(cls)
        new.auto = source["auto"][()]

        new.totals1 = source["totals1"][:]
        new.totals2 = source["totals2"][:]
        if is_legacy_dataset(source):
            new.totals1 = np.transpose(new.totals1)
            new.totals2 = np.transpose(new.totals2)
            new.binning = load_legacy_binning(source)
        else:
            new.binning = Binning.from_hdf(source["binning"])

        return new

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        self.binning.to_hdf(dest.create_group("binning"))
        dest.create_dataset("auto", data=self.auto)

        dest.create_dataset("totals1", data=self.totals1, **HDF_COMPRESSION)
        dest.create_dataset("totals2", data=self.totals2, **HDF_COMPRESSION)

    @property
    def num_patches(self) -> int:
        return self.totals1.shape[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.totals1, other.totals1)
            and np.array_equal(self.totals2, other.totals2)
            and self.auto == other.auto
        )

    def _make_bin_slice(self, item: TypeSliceIndex) -> PatchedTotals:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]
        return type(self)(
            binning, self.totals1[item], self.totals2[item], auto=self.auto
        )

    def _make_patch_slice(self, item: TypeSliceIndex) -> PatchedTotals:
        if isinstance(item, int):
            item = [item]
        return type(self)(
            self.binning, self.totals1[:, item], self.totals2[:, item], auto=self.auto
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
        array = np.einsum("bi,bj->bij", self.totals1, self.totals2)

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
    def zeros(cls, binning: Binning, num_patches: int, *, auto: bool) -> PatchedCounts:
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
    def from_hdf(cls, source: Group) -> PatchedCounts:
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

    def __add__(self, other: Any) -> PatchedCounts:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return type(self)(self.binning, self.counts + other.counts, auto=self.auto)

    def __radd__(self, other: Any) -> PatchedCounts:
        if np.isscalar(other) and other == 0:
            return self  # this convenient when applying sum()

        return self.__add__(other)

    def __mul__(self, other: Any) -> PatchedCounts:
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        return type(self)(self.binning, self.counts * other, auto=self.auto)

    def _make_bin_slice(self, item: TypeSliceIndex) -> PatchedCounts:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]

        return type(self)(binning, self.counts[item], auto=self.auto)

    def _make_patch_slice(self, item: TypeSliceIndex) -> PatchedCounts:
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


class NormalisedCounts(BinwisePatchwiseArray):
    """
    Stores the normalised pair counts in spatial patches from catalogs in a
    correlation measurement.

    This class stores the raw pair counts (:obj:`counts`) and the product of the
    sum of weights (:obj:`totals`) per redshift bin and combination of patch
    pairs. The total of the normalised counts per redshift bin, including
    jackknife samples thereof, can be obtained by calling
    :meth:`sample_patch_sum`. The method computes the sum of pair counts over
    all possible pairs of patches and normalises them by dividing them by the
    product of the sum of weights from catalog 1 and 2.

    Implements comparison with the ``==`` operator, addition of counts with the
    ``+``/``+=`` operator and scalar multiplication of the counts with the ``*``
    operator.

    Args:
        counts:
            Container of correlation pair counts.
        totals:
            Container of sum of weights in patches of catalogs 1 and 2.
    """

    __slots__ = ("counts", "totals")

    counts: PatchedCounts
    """Container of correlation pair counts."""
    totals: PatchedTotals
    """Container of sum of weights in patches of catalogs 1 and 2."""

    def __init__(self, counts: PatchedCounts, totals: PatchedTotals) -> None:
        if counts.num_patches != totals.num_patches:
            raise ValueError("number of patches of 'count' and total' does not match")
        if counts.num_bins != totals.num_bins:
            raise ValueError("number of bins of 'count' and total' does not match")

        self.counts = counts
        self.totals = totals

    @classmethod
    def from_hdf(cls, source: Group) -> NormalisedCounts:
        name = "count" if is_legacy_dataset(source) else "counts"
        counts = PatchedCounts.from_hdf(source[name])

        name = "total" if is_legacy_dataset(source) else "totals"
        totals = PatchedTotals.from_hdf(source[name])

        return cls(counts=counts, totals=totals)

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)

        group = dest.create_group("counts")
        self.counts.to_hdf(group)

        group = dest.create_group("totals")
        self.totals.to_hdf(group)

    @property
    def binning(self) -> Binning:
        return self.counts.binning

    @property
    def auto(self) -> bool:
        return self.counts.auto

    @property
    def num_patches(self) -> int:
        return self.counts.num_patches

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self.counts.is_compatible(other.counts, require=require)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.counts == other.counts and self.totals == other.totals

    def __add__(self, other: Any) -> NormalisedCounts:
        if not isinstance(other, type(self)):
            return NotImplemented

        if self.totals != other.totals:
            raise ValueError("totals must be identical for operation")
        return type(self)(self.counts + other.counts, self.totals)

    def __radd__(self, other: Any) -> NormalisedCounts:
        if np.isscalar(other) and other == 0:
            return self  # allows using sum() on an iterable of NormalisedCounts

        return self.__add__(other)

    def __mul__(self, other: Any) -> NormalisedCounts:
        return type(self)(self.count * other, self.total)

    def _make_bin_slice(self, item: TypeSliceIndex) -> NormalisedCounts:
        counts = self.counts.bins[item]
        totals = self.totals.bins[item]
        return type(self)(counts, totals)

    def _make_patch_slice(self, item: TypeSliceIndex) -> NormalisedCounts:
        counts = self.counts.patches[item]
        totals = self.totals.patches[item]
        return type(self)(counts, totals)

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
        counts = self.counts.get_array()
        totals = self.totals.sample_patch_sum()
        return counts / totals.data

    def sample_patch_sum(self) -> SampledData:
        counts = self.counts.sample_patch_sum()
        totals = self.totals.sample_patch_sum()

        data = counts.data / totals.data
        samples = counts.samples / totals.samples
        return SampledData(self.binning, data, samples)
