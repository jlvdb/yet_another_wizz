import multiprocessing
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from Nz_Fitting.data import RedshiftData, RedshiftDataBinned
from Nz_Fitting.fitting import FitResult


def nz_model_summed_bins(nz_data, *params, check_binning=True, **kwargs):
    """
    nz_data : BinnedRedshiftData
    *params : array_like, curve_fit parameter list -> parameters of bias model
    check_binning : bool, assert that all redshift sampling points are the same
    weights : array_like, one weight per bin, sum normalized to unity
    bias_model : PowerLawBias, the bias model to fit to the data
    """
    master = nz_data.getMaster()
    bins = nz_data.getBins()
    weights = kwargs["weights"]
    bias_model = kwargs["bias_model"]
    # get the redshift sampling end evaluate the redshift binning
    if check_binning:
        nz_data.assertEqual(nz_data.z())  # same redshift binning everywhere
    z = master.z()
    bias = bias_model(z, *params)
    # compute the normalization after dividing the master n(z) by the bias
    norm_factor = np.trapz(master.n() / bias, x=z)
    # compute the weighted sum of the bias corrected and re-normalized bins
    summed_nz = np.empty_like(master.n())
    for nz, weight in zip(bins, weights):
        nz_debiased = nz.n() / bias
        nz_debiased /= np.trapz(nz_debiased, x=z)
        summed_nz += nz_debiased * weight
    # model for the master data
    return bias * norm_factor * summed_nz


def fit_bias_realisation(
        *args, draw_sample=False, check_binning=True, **kwargs):
    """
    *args : array_like, curve_fit parameter list -> parameters of bias model
    draw_sample : bool, whether to draw/get a realisation or not
    nz_data : BinnedRedshiftData
    weights : array_like, one weight per bin, sum normalized to unity
    bias_model : PowerLawBias, the bias model to fit to the data
    check_binning : bool, assert that all redshift sampling points are the same
    **kwargs : parsed to curve_fit
    """
    nz_data = kwargs.pop("nz_data")
    assert(type(nz_data) is RedshiftDataBinned)
    weights = kwargs.pop("weights")
    assert(len(weights) == len(nz_data.getBins()))
    bias_model = kwargs.pop("bias_model")
    # get the data sample to fit
    if draw_sample:
        fit_data = nz_data.getSample(args[0])  # the sample index
    else:
        fit_data = nz_data  # fiducial fit
    # get the data uncertainty
    master = fit_data.getMaster()
    if master.hasCovMat() and master.getCovMatType() != "diagonal":
        sigma = master.getCovMat()
    else:
        sigma = master.dn(concat=True)
    # run the optimizer
    fit_func = partial(
        nz_model_summed_bins, check_binning=check_binning,
        weights=weights, bias_model=bias_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(
            fit_func, fit_data, master.n(concat=True), sigma=sigma,
            p0=bias_model.getParamGuess(),
            bounds=bias_model.getParamBounds(),
            **kwargs)
    # compute the chi squared
    diff_data_model = fit_func(fit_data, *popt) - master.n(concat=True)
    if len(sigma.shape) > 1:  # covariance matrix
        chisq = np.matmul(
            diff_data_model, np.matmul(np.linalg.inv(sigma), diff_data_model))
    else:  # standard errors
        chisq = np.sum((diff_data_model / sigma)**2)
    return popt, chisq


def fit_bias(
        nz_data, weights, bias_model, threads=None, check_binning=True,
        **kwargs):
    """
    nz_data : BinnedRedshiftData
    weights : array_like, one weight per bin, sum normalized to unity
    bias_model : PowerLawBias, the bias model to fit to the data
    threads : int, number of threads to use
    check_binning : bool, assert that all redshift sampling points are the same
    **kwargs : parsed to curve_fit
    """
    label_dict = OrderedDict(zip(
        bias_model.getParamNames(),
        bias_model.getParamlabels()))
    guess = bias_model.getParamGuess()
    bounds = bias_model.getParamBounds()
    # get the best fit parameters
    pbest, chisq = fit_bias_realisation(
        draw_sample=False, nz_data=nz_data, check_binning=check_binning,
        weights=weights, bias_model=bias_model, **kwargs)
    bias_model.setParamGuess(pbest)
    pbest_dict = OrderedDict(zip(label_dict.keys(), pbest))
    # resample data points for each fit to estimate parameter covariance
    if threads is None:
        threads = multiprocessing.cpu_count()
    # generate samples ahead of time if necessecary
    if not nz_data.hasSamples():
        n_samples = 1000
        # generate sample for the bins
        for i, nz_bin in enumerate(nz_data.iterBins(), 1):
            print("resampling bin %d" % i)
            samples = np.empty((n_samples, nz_bin.len(all=True)))
            for n, nz in enumerate(nz_bin.iterSamples(n_samples)):
                samples[n] = nz.n(all=True)
            nz_bin.setSamples(samples)
        # generate samples for the master sample
        nz_master = nz_data.getMaster()
        print("resampling master data")
        samples = np.empty((n_samples, nz_master.len(all=True)))
        for n, nz in enumerate(nz_master.iterSamples(n_samples)):
            samples[n] = nz.n(all=True)
        nz_master.setSamples(samples)
        assert(nz_data.hasSamples())
    # get the number of samples to use
    n_samples = nz_data.getSampleNo()
    threads = min(threads, n_samples)
    chunksize = n_samples // threads + 1  # optmizes the workload
    threaded_fit = partial(
        fit_bias_realisation, draw_sample=True, nz_data=nz_data,
        check_binning=check_binning, weights=weights, bias_model=bias_model,
        **kwargs)
    # run in parallel threads
    with multiprocessing.Pool(threads) as pool:
        param_samples = pool.map(
            threaded_fit, range(n_samples), chunksize=chunksize)
    param_samples_dict = OrderedDict(
        zip(label_dict.keys(), np.transpose([
            p for p, chi in param_samples])))
    # collect the best fit data
    bestfit = FitResult(
        pbest_dict, param_samples_dict, label_dict,
        nz_data.len(concat=True) - len(pbest), chisq)
    return bestfit


def bias_evaluate_bestfit(z, bias_model, best_fit):
    """
    z : array_like, redshift sampling
    bias_model : PowerLawBias, the bias model fitted to the data
    best_fit : FitResult, best fit parameters and samples
    """
    param = best_fit.paramBest()
    param_samples = best_fit.paramSamples()
    # compute the fiducial
    bias = bias_model(z, *param)
    # compute the realisations
    bias_samples = np.empty((len(param_samples), len(z)))
    for i, param in enumerate(param_samples):
        bias_samples[i] = bias_model(z, *param)
    # pack in redshift container
    dummy_error = np.ones_like(bias)  # computed from samples later
    bias_data = RedshiftData(z, bias, dummy_error)
    bias_data.setSamples(bias_samples)
    return bias_data


def apply_bias_fit(nz_original, bias_model, best_fit):
    """
    nz_original : RedshiftDataBinned, original (bias) n(z)'s
    bias_model : PowerLawBias, the bias model fitted to the data
    best_fit : FitResult, best fit parameters and samples
    """
    # evaluate the bias model
    z = nz_original.getMaster().z(all=True)
    bias_data = bias_evaluate_bestfit(z, bias_model, best_fit)
    # apply to the original data
    nz_debiased = []
    for nz_bin in nz_original.iterData():
        mask = ~nz_bin.mask()
        z_bin = nz_bin.z(all=True)
        # fiducial n(z)
        nz_data = nz_bin.n(all=True) / bias_data.n(all=True)
        norm_data = np.trapz(nz_data[mask], x=z_bin[mask])
        nz_data /= norm_data
        # n(z) samples
        nz_samples = \
            nz_bin.getSamples(all=True) / bias_data.getSamples(all=True)
        norm_samples = np.empty(len(nz_samples))
        for i, sample in enumerate(nz_samples):
            sample_mask = mask & np.isfinite(sample)
            norm_samples[i] = \
                np.trapz(sample[sample_mask], x=z_bin[sample_mask])
            nz_samples[i] /= norm_samples[i]
        # pack in redshift container
        dummy_error = np.ones_like(nz_data)  # computed from samples later
        nz_data = RedshiftData(z_bin, nz_data, dummy_error)
        nz_data.setSamples(nz_samples)
        nz_debiased.append(nz_data)
    nz_debiased = RedshiftDataBinned(nz_debiased[:-1], nz_debiased[-1])
    # take the latest normalisation factors to rescale the bias data
    bias_rescaled = RedshiftData(  # errors computed from samples later
        z, bias_data.n(all=True) * norm_data, bias_data.dn(all=True))
    bias_rescaled.setSamples(
        bias_data.getSamples(all=True) * norm_samples[:, np.newaxis])
    return bias_rescaled, nz_debiased
