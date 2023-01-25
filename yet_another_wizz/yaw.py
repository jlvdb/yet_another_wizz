from __future__ import annotations

from typing import Any

from yet_another_wizz.core.catalog import CatalogBase, PatchLinkage
from yet_another_wizz.core.config import Configuration
from yet_another_wizz.core.correlation import CorrelationFunction
from yet_another_wizz.core.utils import Timed, TypeScaleKey


def _create_dummy_counts(
    counts: Any | dict[TypeScaleKey, Any]
) -> dict[TypeScaleKey, None]:
    if isinstance(counts, dict):
        dummy = {scale_key: None for scale_key in counts}
    else:
        dummy = None
    return dummy


def autocorrelate(
    config: Configuration,
    data: CatalogBase,
    random: CatalogBase,
    compute_rr: bool = True
) -> CorrelationFunction | dict[TypeScaleKey, CorrelationFunction]:
    """
    Compute the angular autocorrelation amplitude in bins of redshift. Requires
    object redshifts.
    """
    print(f"autocorrelating")
    linkage = PatchLinkage.from_setup(config, random)
    with Timed("counting data-data pairs"):
        DD = data.correlate(
            config, binned=True, linkage=linkage)
    with Timed("counting data-random pairs"):
        DR = data.correlate(
            config, binned=True, other=random, linkage=linkage)
    if compute_rr:
        with Timed("counting random-random pairs"):
            RR = random.correlate(config, binned=True, linkage=linkage)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrelationFunction(dd=DD[scale], dr=DR[scale], rr=RR[scale])
            for scale in DD}
    else:
        result = CorrelationFunction(dd=DD, dr=DR, rr=RR)
    return result


def crosscorrelate(
    config: Configuration,
    reference: CatalogBase,
    unknown: CatalogBase,
    *,
    ref_rand: CatalogBase | None = None,
    unk_rand: CatalogBase | None = None
) -> CorrelationFunction | dict[TypeScaleKey, CorrelationFunction]:
    """
    Compute the angular cross-correlation amplitude in bins of redshift with
    another catalogue instance. Requires object redshifts in this catalogue
    instance.
    """
    compute_dr = unk_rand is not None
    compute_rd = ref_rand is not None
    compute_rr = compute_dr and compute_rd
    if not compute_rd and not compute_dr:
        raise ValueError("no randoms provided")

    print("crosscorrelating")
    linkage = PatchLinkage.from_setup(config, unknown)
    with Timed("counting data-data pairs"):
        DD = reference.correlate(
            config, binned=False, other=unknown, linkage=linkage)
    if compute_dr:
        with Timed("counting data-random pairs"):
            DR = reference.correlate(
                config, binned=False, other=unk_rand, linkage=linkage)
    else:
        DR = _create_dummy_counts(DD)
    if compute_rd:
        with Timed("counting random-data pairs"):
            RD = ref_rand.correlate(
                config, binned=False, other=unknown, linkage=linkage)
    else:
        RD = _create_dummy_counts(DD)
    if compute_rr:
        with Timed("counting random-random pairs"):
            RR = ref_rand.correlate(
                config, binned=False, other=unk_rand, linkage=linkage)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrelationFunction(
                dd=DD[scale], dr=DR[scale], rd=RD[scale], rr=RR[scale])
            for scale in DD}
    else:
        result = CorrelationFunction(dd=DD, dr=DR, rd=RD, rr=RR)
    return result
