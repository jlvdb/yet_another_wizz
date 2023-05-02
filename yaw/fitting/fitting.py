from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from yaw.correlation import HistogramData
from yaw.utils import cov_from_samples

from yaw.fitting.models import ModelEnsemble, ShiftModel
from yaw.fitting.optmize import Optimizer
from yaw.fitting.priors import Prior, GaussianPrior
from yaw.fitting.samples import MCSamples

if TYPE_CHECKING:  # pragma: no cover
    from yaw.correlation import RedshiftData


def shift_fit(
    data: Sequence[RedshiftData],
    model: Sequence[HistogramData],
    *,
    covariance: str = "var",
    nwalkers: int = None,
    max_steps: int = 10000,
    optimise: bool = True
) -> MCSamples:
    # verify that data and model have matching binning etc.
    for i, (bin_data, bin_model) in enumerate(zip(data, model), 1):
        if not bin_data.is_compatible(bin_model):
            raise ValueError(
                "'data' and 'model' are not compatible in binning, spatial "
                f"patches, or resampling method for the {i}-th entry")

    # build the joint data vector
    data_all = np.concatenate([bin_data.data for bin_data in data])
    cov_mat_all = cov_from_samples(
        [bin_data.samples for bin_data in data],
        method=bin_data.method, kind=covariance)
    # mask bad values
    var_all = np.diag(cov_mat_all)
    mask = np.isfinite(data_all) & np.isfinite(var_all) & (var_all > 0.0)

    # build the models
    fit_models = [
        ShiftModel(bin_model.edges, bin_model.data, bin_data.edges)
        for bin_model, bin_data in zip(model, data)]
    fit_model = ModelEnsemble(fit_models)
    full_model = Optimizer(
        data_all, np.linalg.inv(cov_mat_all), fit_model)
    full_model.set_mask(mask)

    # add the priors
    priors: list[Prior] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bin_data, bin_model in zip(data, model):
            rebinned = bin_model.rebin(bin_data.edges)
            norm = np.trapz(rebinned.data, x=bin_data.mids)
            prior_log_amp = GaussianPrior(-np.log10(norm), 0.5)
            prior_dz = GaussianPrior(0.0, 0.015)
            priors.extend([prior_log_amp, prior_dz])
    full_model.set_priors(priors)

    samples = full_model.run_mcmc(nwalkers=nwalkers, max_steps=max_steps)
    if optimise:
        best, chisq = full_model.run_fitting(samples.best().to_numpy())
        samples.set_best(best, chisq)
    return samples
