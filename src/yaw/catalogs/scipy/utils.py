from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from yaw.catalogs import PatchLinkage
from yaw.catalogs.scipy.patches import PatchCatalog
from yaw.config import Configuration, ResamplingConfig
from yaw.core.containers import PatchCorrelationData, PatchIDs
from yaw.correlation.paircounts import (
    NormalisedCounts,
    PatchedCount,
    PatchedTotal,
    pack_results,
)
from yaw.redshifts import HistData

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalogs.scipy.catalog import ScipyCatalog

__all__ = [
    "get_patch_list",
    "count_pairs_patches",
    "merge_pairs_patches",
    "count_histogram_patch",
    "merge_histogram_patches",
]


def get_patch_list(
    catalog1: ScipyCatalog,
    catalog2: ScipyCatalog | None,
    config: Configuration,
    linkage: PatchLinkage | None,
    auto: bool,
) -> tuple[list[PatchCatalog], list[PatchCatalog]]:
    """Generate a two lists of patch pairs to correlate.

    Generate the listing from two catalogs either from a given linkage or from
    a newly constructed one.

    Args:
        catalog1 (:obj:`yaw.catalogs.scipy.ScipyCatalog`):
            The first input data catalogue.
        catalog2 (:obj:`yaw.catalogs.scipy.ScipyCatalog`, :obj:`None`):
            The second input data catalogue, can be `None`.
        config (:obj:`yaw.config.Configuration`):
            The configuration used for the correlation measurement.
        linkage (:obj:`~yaw.catalogs.linkage.PatchLinkage`, :obj:`None`):
            Linkage object that defines with patches must be correlated for
            a given scales and which patch combinations can be skipped.
        auto (:obj:`bool`):
            Whether to generate patch pairs for an autocorrelation.

    Returns:
        Two lists, one containing the patches from the first catalogue that are
        paired with the ones in the second list from the second catalogues.
    """
    if linkage is None:
        if not auto and len(catalog2) > len(catalog1):
            cat_for_linkage = catalog2
        else:
            cat_for_linkage = catalog1
        linkage = PatchLinkage.from_setup(config, cat_for_linkage)
    return linkage.get_patches(catalog1, catalog2, config.backend.crosspatch)


def count_pairs_patches(
    patch1: PatchCatalog,
    patch2: PatchCatalog,
    config: Configuration,
    bin1: bool = True,
    bin2: bool = False,
) -> PatchCorrelationData:
    """Implementes the pair counting between two patches in bins of redshift.

    Bins the data as needed and builds the KDTrees for the pair finding.
    Converts the physical scales to angles for the given cosmology and redshift
    and counts the pairs. Pairs are recoreded for each set of scales and stored
    in a PatchCorrelationData object.

    Args:
        patch1 (:obj:`yaw.catalogs.scipy.PatchCatalog`):
            The first input patch catalogue.
        patch2 (:obj:`yaw.catalogs.scipy.PatchCatalog`):
            The second input patch catalogue.
        config (:obj:`yaw.config.Configuration`):
            The configuration used for the correlation measurement.
        bin1 (:obj:`bool`):
            Whether to apply binning to the first patch.
        bin2 (:obj:`bool`):
            Whether to apply binning to the second patch.

    Returns:
        A container containing the patch IDs, number of objects from both
        patches and the number of pair counts, each in bins of redshift.
    """
    scales = list(config.scales)
    z_bins = config.binning.zbins
    z_intervals = pd.IntervalIndex.from_breaks(z_bins)
    # build trees
    patch1.load(use_threads=False)
    if bin1:
        trees1 = [patch.get_tree() for _, patch in patch1.iter_bins(z_bins)]
    else:
        trees1 = itertools.repeat(patch1.get_tree())
    patch2.load(use_threads=False)
    if bin2:
        trees2 = [patch.get_tree() for _, patch in patch2.iter_bins(z_bins)]
    else:
        trees2 = itertools.repeat(patch2.get_tree())
    # count pairs, iterate through the bins and count pairs between the trees
    counts = np.empty((len(scales), len(z_intervals)))
    totals1 = np.empty(len(z_intervals))
    totals2 = np.empty(len(z_intervals))
    for i, (intv, tree1, tree2) in enumerate(zip(z_intervals, trees1, trees2)):
        # if bin1 is False and bin2 is False, these will still give different
        # counts since the angle for scales is chaning
        angles = [scale.to_radian(intv.mid, config.cosmology) for scale in scales]
        counts[:, i] = tree1.count(
            tree2,
            scales=angles,
            dist_weight_scale=config.scales.rweight,
            weight_res=config.scales.rbin_num,
        )
        totals1[i] = tree1.total
        totals2[i] = tree2.total
    counts = {str(scale): count for scale, count in zip(scales, counts)}
    return PatchCorrelationData(
        patches=PatchIDs(patch1.id, patch2.id),
        totals1=totals1,
        totals2=totals2,
        counts=counts,
    )


def merge_pairs_patches(
    patch_datasets: Iterable[PatchCorrelationData],
    config: Configuration,
    n_patches: int,
    auto: bool,
) -> NormalisedCounts | dict[str, NormalisedCounts]:
    """Merge pair counts from patch pairs into a pair count container.

    Args:
        patch_datasets (obj:`Iterable[PatchCorrelationData]`):
            An iterable containing pair counts measured from pairs of patches.
        config (:obj:`yaw.config.Configuration`):
            The configuration used for the correlation measurement.
        n_patches (:obj:`int`):
            The total number of patches in both catalogs.
        auto (:obj:`bool`):
            Whether the pair counts are from an autocorrelation measurement.

    Returns:
        A :obj:`~yaw.correlation.paircounts.NormalisedCounts` instance if a
        single measurement scale is used, otherwise a dictionary of scales.
    """
    binning = pd.IntervalIndex.from_breaks(config.binning.zbins)
    n_bins = len(binning)
    # set up data to repack task results from [ids->scale] to [scale->ids]
    totals1 = np.zeros((n_patches, n_bins))
    totals2 = np.zeros((n_patches, n_bins))
    count_dict = {
        str(scale): PatchedCount.zeros(binning, n_patches, auto=auto)
        for scale in config.scales
    }
    # unpack and process the counts for each scale
    for patch_data in patch_datasets:
        id1, id2 = patch_data.patches
        # record total weight per bin, overwriting OK since identical
        totals1[id1] = patch_data.totals1
        totals2[id2] = patch_data.totals2
        # record counts at each scale
        for scale_key, count in patch_data.counts.items():
            if auto and id1 == id2:
                count = count * 0.5  # autocorrelation pairs are counted twice
            count_dict[scale_key].set_measurement((id1, id2), count)
    # collect totals which do not depend on scale
    total = PatchedTotal(binning=binning, totals1=totals1, totals2=totals2, auto=auto)
    return pack_results(count_dict, total)


def count_histogram_patch(
    patch: PatchCatalog, z_bins: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute a histogram of redshifts in a single patch.

    Args:
        patch (:obj:`yaw.catalogs.scipy.PatchCatalog`):
            The input patch catalogue.
        z_bins (:obj:`NDArray[np.float64]`):
            The bin edges including the right-most edge.

    Returns:
        :obj:`NDArray[np.float64]`:
            Counts in the provided redshift bins.
    """
    is_loaded = patch.is_loaded()
    patch.load()
    counts, _ = np.histogram(patch.redshifts, z_bins, weights=patch.weights)
    if not is_loaded:
        patch.unload()
    return counts


def merge_histogram_patches(
    hist_counts: NDArray[np.float64],
    z_bins: NDArray[np.float64],
    sampling_config: ResamplingConfig | None = None,
) -> HistData:
    """Merge redshift histogram from patches into a histogram data container.

    Args:
        hist_counts (:obj:`NDArray[np.float64]`):
            A two-dimensional array with histogram counts with shape
            `(n_patches, n_bins)`.
        z_bins (:obj:`NDArray[np.float64]`):
            The bin edges including the right-most edge.
        sampling_config: (:obj:`yaw.config.ResamplingConfig`, optional):
            Specify the resampling method and its configuration.

    Returns:
        :obj:`yaw.redshifts.HistData`:
            Histogram data with samples and covaraiance estimate.
    """
    if sampling_config is None:
        sampling_config = ResamplingConfig()  # default values
    binning = pd.IntervalIndex.from_breaks(z_bins)
    patch_idx = sampling_config.get_samples(len(hist_counts))
    nz_data = hist_counts.sum(axis=0)
    nz_samp = np.sum(hist_counts[patch_idx], axis=1)
    return HistData(
        binning=binning,
        data=nz_data,
        samples=nz_samp,
        method=sampling_config.method,
    )
