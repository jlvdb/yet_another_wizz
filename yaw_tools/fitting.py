import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def nz_model(z, *params, bias_model=None, nz_bins=None, weights=None):
    assert(bias_model is not None)
    assert(nz_bins is not None)
    assert(weights is not None)
    bias = bias_model(z, params[0])
    # compute the normalizations of each bin such that we can renormalize to
    # this after dividing out the bias model
    norms = [np.trapz(nz.n, x=nz.z) for nz in nz_bins]
    # compute the weighted sum of the bins while applying the bias model
    nz_sum = np.zeros_like(z)
    for weight, nz, norm in zip(weights, nz_bins, norms):
        nz_bin = weight * nz.n / bias  # weighted, bias corrected bin n(z)
        #nz_bin *= norm / np.trapz(nz_bin, x=z)  # restore original norm
        nz_bin = weight * nz.n / np.trapz(nz.n, x=nz.z)
        nz_sum += nz_bin
    # on the other side, the full sample, we would need to compute the
    # re-normalized n_tot / bias as well, therefore multiply weighted sum by
    # bias and the free normalization fit parameter
    nz_full_model = nz_sum * bias * params[1]
    return nz_full_model


def fit_bias(bin_container, weights):
    bias_model = PowerLawBias()
    # unpack the n(z) bin container
    nzs = bin_container.split()
    nz_bins = nzs[:-1]
    nz_full = nzs[-1]
    full_norm = np.trapz(nz_full.n, x=nz_full.z)
    fit_model = partial(
        nz_model, bias_model=bias_model, nz_bins=nz_bins, weights=weights)
    popt, pcov = curve_fit(
        fit_model, nz_full.z, nz_full.n / full_norm,
        sigma=nz_full.dn / full_norm,
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
