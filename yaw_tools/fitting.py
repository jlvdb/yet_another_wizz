from functools import partial

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from Nz_Fitting import PowerLawBias


def nz_model(z, *params, bias_model=None, nz_bins=None, weights=None, norm=1):
    assert(bias_model is not None)
    assert(nz_bins is not None)
    assert(weights is not None)
    bias = bias_model(z, params[0])
    # compute the normalizations of each bin such that we can renormalize to
    # this after dividing out the bias model
    norms = {key: np.trapz(nz.n, x=nz.z) for key, nz in nz_bins.items()}
    # compute the weighted sum of the bins while applying the bias model
    nz_sum = np.zeros_like(z)
    for key, nz in nz_bins.items():
        nz_debiased = nz.n / bias  # weighted, bias corrected bin n(z)
        nz_debiased /= np.trapz(nz_debiased, x=nz.z)
        nz_sum += weights[key] * nz_debiased
    # on the other side, the full sample, we would need to compute the
    # re-normalized n_tot / bias as well, therefore multiply weighted sum by
    # bias and the free normalization fit parameter
    nz_full_model = nz_sum * bias * params[1]
    return nz_full_model * norm


def fit_bias(nz_bins, nz_full, weights):
    bias_model = PowerLawBias()
    full_norm = np.trapz(nz_full.n, x=nz_full.z)
    fit_model = partial(
        nz_model, bias_model=bias_model, nz_bins=nz_bins, weights=weights)
    if nz_full.cov is not None:
        sigmas = nz_full.cov
    else:
        sigmas = nz_full.dn
    popt, pcov = curve_fit(
        fit_model, nz_full.z, nz_full.n, sigma=nz_full.dn,
        p0=[0.0, 1.0])  # alpha, normalisation full sample rel. to weighted sum
    return popt, pcov


class model_from_n_z(object):

    def __init__(self, z, y):
        self.p_z = y / np.trapz(y, x=z)
        self.pdf = interp1d(z, self.p_z, fill_value="extrapolate")
        self.c_z = cumtrapz(self.p_z, x=z, initial=0.0)
        self.cdf = interp1d(z, self.c_z, fill_value="extrapolate")

    def rebin(self, z_edges):
        P_edges = self.cdf(z_edges)
        return np.diff(P_edges) / np.diff(z_edges)
