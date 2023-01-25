from __future__ import annotations

from typing import Any

from yet_another_wizz.core.catalog import CatalogBase, PatchLinkage
from yet_another_wizz.core.config import Configuration
from yet_another_wizz.core.correlation import CorrelationFunction
from yet_another_wizz.core.utils import Timed, TypeScaleKey


def _create_dummy_counts(
    counts_dict: dict[TypeScaleKey, Any]
) -> dict[TypeScaleKey, None]:
    return {scale_key: None for scale_key in counts_dict}


def autocorrelate(
    config: Configuration,
    data: CatalogBase,
    random: CatalogBase,
    compute_rr: bool = True
) -> dict[TypeScaleKey, CorrelationFunction]:
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

    corrfuncs = {
        scale_key: CorrelationFunction(
            dd=DD[scale_key],
            dr=DR[scale_key],
            rr=RR[scale_key])
        for scale_key in DD}
    return corrfuncs


def crosscorrelate(
    config: Configuration,
    reference: CatalogBase,
    unknown: CatalogBase,
    *,
    ref_rand: CatalogBase | None = None,
    unk_rand: CatalogBase | None = None
) -> dict[TypeScaleKey, CorrelationFunction]:
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

    corrfuncs = {
        scale_key: CorrelationFunction(
            dd=DD[scale_key],
            dr=DR[scale_key],
            rd=RD[scale_key],
            rr=RR[scale_key])
        for scale_key in DD}
    return corrfuncs
