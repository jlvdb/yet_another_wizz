from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from yaw.core.math import cov_from_samples
from yaw.redshifts import HistData

from yaw.fitting.models import ModelEnsemble, ShiftModel
from yaw.fitting.optimize import Optimizer
from yaw.fitting.priors import Prior, GaussianPrior
from yaw.fitting.samples import MCSamples

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from yaw.redshifts import RedshiftData


def shift_fit(
    data: Sequence[RedshiftData],
    model: Sequence[HistData],
    *,
    covariance: NDArray | str = "var",
    nwalkers: int = None,
    max_steps: int = 10000,
    dz_priors: Sequence[Prior] | None = None,
    optimise: bool = True
) -> MCSamples:
    # build the joint data vector
    data_all = np.concatenate([bin_data.data for bin_data in data])
    method = data[0].method
    if isinstance(covariance, str):
        cov_mat_all = cov_from_samples(
            [bin_data.samples for bin_data in data],
            method=method, kind=covariance)
    elif covariance.shape != (len(data_all), len(data_all)):
        raise ValueError(
            f"expected covariance with shape {(len(data_all), len(data_all))}, "
            f"but got {covariance.shape}")
    else:
        cov_mat_all = covariance
    # mask bad values
    var_all = np.diag(cov_mat_all)
    mask = np.isfinite(data_all) & np.isfinite(var_all) & (var_all > 0.0)

    # build the models
    fit_models = [
        ShiftModel(bin_model.edges, bin_model.data, bin_data.edges)
        for bin_model, bin_data in zip(model, data)]
    fit_model = ModelEnsemble(fit_models)
    full_model = Optimizer(data_all, cov_mat_all, fit_model)
    full_model.set_mask(mask)

    # add the priors
    priors: list[Prior] = []
    for i, (bin_data, bin_model) in enumerate(zip(data, model)):
        rebinned = bin_model.rebin(bin_data.edges)
        norm = np.trapz(rebinned.data, x=bin_data.mids)
        prior_log_amp = GaussianPrior(-np.log10(norm), 0.5)
        if dz_priors is None:
            prior_dz = GaussianPrior(0.0, 0.015)
        else:
            prior_dz = dz_priors[i]
        priors.extend([prior_log_amp, prior_dz])
    full_model.set_priors(priors)

    samples = full_model.run_mcmc(nwalkers=nwalkers, max_steps=max_steps)
    if optimise:
        best, chisq = full_model.run_fitting(samples.best().to_numpy())
        samples.set_best(best, chisq)
    return samples
