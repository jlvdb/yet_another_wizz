from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.catalog.kdtree import PatchLinkage
from yaw.core.utils import TimedLog
from yaw.correlation.corrfuncs import CorrFunc

if TYPE_CHECKING:  # pragma: no cover
    from yaw.catalog import Catalog
    from yaw.config import Configuration

__all__ = ["autocorrelate", "crosscorrelate"]


logger = logging.getLogger(__name__)


def _create_dummy_counts(counts: Any | dict[str, Any]) -> dict[str, None]:
    """Duplicate a the return values of
    :meth:`yaw.catalog.Catalog.correlate`, but replace the :obj:`CorrFunc`
    instances by :obj:`None`."""
    if isinstance(counts, dict):
        dummy = {scale_key: None for scale_key in counts}
    else:
        dummy = None
    return dummy


class PatchError(Exception):
    pass


def _check_patch_centers(catalogues: Sequence[Catalog]) -> None:
    """Check whether the patch centers of a set of data catalogues are seperated
    by no more than the radius of the patches."""
    refcat = catalogues[0]
    for cat in catalogues[1:]:
        if refcat.n_patches != cat.n_patches:
            raise PatchError("number of patches does not agree")
        ref_coord = refcat.centers.to_sky()
        cat_coord = cat.centers.to_sky()
        dist = ref_coord.distance(cat_coord)
        if np.any(dist.values > refcat.radii.values):
            raise PatchError("the patch centers are inconsistent")


def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    linkage: PatchLinkage | None = None,
    compute_rr: bool = True,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    """Compute an angular autocorrelation function in bins of redshift.

    The correlation is measured on fixed physical scales that are converted to
    angles for each redshift bin. All parameters (binning, scales, etc.) are
    bundled in the input configuration, see :mod:`yaw.config`.

    .. Note::
        Both the data and random catalogue require redshift point estimates.

    Args:
        config (:obj:`~yaw.config.Configuration`):
            Provides all major run parameters, such as scales, binning, and for
            the correlation measurement backend.
        data (:obj:`~yaw.catalog.Catalog`):
            The data sample catalogue.
        random (:obj:`~yaw.catalog.Catalog`):
            Random catalogue for the data sample.

    Keyword Args:
        linkage (:obj:`~yaw.catalog.PatchLinkage`, optional):
            Provide a linkage object that determines which spatial patches must
            be correlated given the measurement scales. Ensures consistency
            when measuring correlations repeatedly for a fixed set of input
            catalogues. Generated automatically by default.
        compute_rr (:obj:`bool`):
            Whether the random-random (RR) pair counts are computed.
        progress (:obj:`bool`):
            Display a progress bar.

    Returns:
        :obj:`CorrFunc` or :obj:`dict[str, CorrFunc]`:
            Container that holds the measured pair counts, or a dictionary of
            containers if multiple scales are configured. Dictionary keys have a
            ``kpcXXtXX`` pattern, where ``XX`` are the lower and upper scale
            limit as integers, in kpc (see :obj:`yaw.core.cosmology.Scale`).
    """
    _check_patch_centers([data, random])
    scales = config.scales.as_array()
    logger.info(
        "running autocorrelation (%i scales, %.0f<r<=%.0fkpc)",
        len(scales),
        scales.min(),
        scales.max(),
    )
    if linkage is None:
        linkage = PatchLinkage(config, random)
    kwargs = dict(linkage=linkage, progress=progress)
    logger.debug("scheduling DD, DR" + (", RR" if compute_rr else ""))
    with TimedLog(logger.info, "counting data-data pairs"):
        DD = data.correlate(config, binned=True, **kwargs)
    with TimedLog(logger.info, "counting data-rand pairs"):
        DR = data.correlate(config, binned=True, other=random, **kwargs)
    if compute_rr:
        with TimedLog(logger.info, "counting rand-rand pairs"):
            RR = random.correlate(config, binned=True, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrFunc(dd=DD[scale], dr=DR[scale], rr=RR[scale]) for scale in DD
        }
    else:
        result = CorrFunc(dd=DD, dr=DR, rr=RR)
    return result


def crosscorrelate(
    config: Configuration,
    reference: Catalog,
    unknown: Catalog,
    *,
    ref_rand: Catalog | None = None,
    unk_rand: Catalog | None = None,
    linkage: PatchLinkage | None = None,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    """Compute an angular crosscorrelation function in bins of redshift.

    The correlation is measured on fixed physical scales that are converted to
    angles for each redshift bin. All parameters (binning, scales, etc.) are
    bundled in the input configuration, see :mod:`yaw.config`.

    At least one random catalogue (either for the reference or the unknown
    sample) must be provided, which will either trigger counting the DR
    (reference-random) or RD (random-unknown) pair counts. If both random
    catalogues are provided, the random-random pairs (RR) are counted as well,
    this is equivalent to enabling the ``compute_rr`` parameter in
    :func:`autocorrelate`.

    .. Note::
        The reference catalogue requires redshift point estimates. If the
        reference random cataloge is provided, it also requires redshifts.

    Args:
        config (:obj:`~yaw.config.Configuration`):
            Provides all major run parameters.
        reference (:obj:`yaw.catalog.Catalog`):
            The reference sample.
        unknown (:obj:`yaw.catalog.Catalog`):
            The sample with unknown redshift distribution.

    Keyword Args:
        ref_rand (:obj:`yaw.catalog.Catalog`, optional):
            Random catalog for the reference sample, requires redshifts
            configured.
        unk_rand (:obj:`yaw.catalog.Catalog`, optional):
            Random catalog for the unknown sample.
        linkage (:obj:`yaw.catalog.PatchLinkage`, optional):
            Provide a linkage object that determines which spatial patches must
            be correlated given the measurement scales. Ensures consistency
            when measuring multiple correlations, otherwise generated
            automatically.
        progress (:obj:`bool`):
            Display a progress bar.

    Returns:
        :obj:`CorrFunc` or :obj:`dict[str, CorrFunc]`:
            Container that holds the measured pair counts, or a dictionary of
            containers if multiple scales are configured. Dictionary keys have a
            ``kpcXXtXX`` pattern, where ``XX`` are the lower and upper scale
            limit as integers, in kpc (see :obj:`yaw.core.cosmology.Scale`).
    """
    compute_dr = unk_rand is not None
    compute_rd = ref_rand is not None
    compute_rr = compute_dr and compute_rd
    # make sure that the patch centers are consistent
    all_cats = [reference, unknown]
    if compute_dr:
        all_cats.append(unk_rand)
    if compute_rd:
        all_cats.append(ref_rand)
    _check_patch_centers(all_cats)

    scales = config.scales.as_array()
    logger.info(
        "running crosscorrelation (%i scales, %.0f<r<=%.0fkpc)",
        len(scales),
        scales.min(),
        scales.max(),
    )
    if linkage is None:
        linkage = PatchLinkage(config, unknown)
    logger.debug(
        "scheduling DD"
        + (", DR" if compute_dr else "")
        + (", RD" if compute_rd else "")
        + (", RR" if compute_rr else "")
    )
    kwargs = dict(linkage=linkage, progress=progress)
    with TimedLog(logger.info, "counting data-data pairs"):
        DD = reference.correlate(config, binned=False, other=unknown, **kwargs)
    if compute_dr:
        with TimedLog(logger.info, "counting data-rand pairs"):
            DR = reference.correlate(config, binned=False, other=unk_rand, **kwargs)
    else:
        DR = _create_dummy_counts(DD)
    if compute_rd:
        with TimedLog(logger.info, "counting rand-data pairs"):
            RD = ref_rand.correlate(config, binned=False, other=unknown, **kwargs)
    else:
        RD = _create_dummy_counts(DD)
    if compute_rr:
        with TimedLog(logger.info, "counting rand-rand pairs"):
            RR = ref_rand.correlate(config, binned=False, other=unk_rand, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrFunc(dd=DD[scale], dr=DR[scale], rd=RD[scale], rr=RR[scale])
            for scale in DD
        }
    else:
        result = CorrFunc(dd=DD, dr=DR, rd=RD, rr=RR)
    return result
