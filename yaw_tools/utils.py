import argparse
import os

import numpy as np
from numpy import ma

from Nz_Fitting import format_variable, RedshiftData, RedshiftDataBinned

from .folders import (DEFAULT_EXT_BOOT, DEFAULT_EXT_COV, DEFAULT_EXT_DATA,
                      binname, ScaleFolder)


DEFAULT_CAT_EXT = 1
DEFAULT_REGION_NO = 7
DEFAULT_RESAMPLING = 10
DEFAULT_PAIR_WEIGHTING = "Y"


def guess_bin_order(bin_keys):

    def get_zmin(key):
        zmin, zmax = [float(z) for z in key.split("z")]
        return zmin

    def get_zmax(key):
        zmin, zmax = [float(z) for z in key.split("z")]
        return zmax

    keys_sorted = sorted(bin_keys, key=get_zmin)
    # one bin could be the master sample
    zmin = np.inf
    zmax = 0.0
    for key in keys_sorted:
        zmin = min(zmin, get_zmin(key))
        zmax = max(zmax, get_zmax(key))
    master_key = None
    for i, key in enumerate(keys_sorted):
        if zmin == get_zmin(key) and zmax == get_zmax(key):
            master_key = keys_sorted.pop(i)
            break
    if master_key is not None:
        keys_sorted.append(master_key)
    return keys_sorted


def TypeNone(type):
    """Test input value for being 'type', but also accepts None as default

    Arguments:
        type [type]:
            specifies the type (int or float) the input argument has to obey
    Returns:
        type_text [function]:
            function that takes 'value' as argument and tests, if it is either
            None or of type 'type'
    """
    def type_test(value):
        if value is None:
            return value
        else:
            strtype = "float" if type == float else "int"
            try:
                return type(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "invalid %s value: '%s'" % (strtype, value))
    return type_test  # closure being of fixed type


def nancov(bootstraps, ddof=0, rowvar=False):
    # mask infinite values
    mask = np.isnan(bootstraps)
    masked_boots = ma.array(bootstraps, mask=mask)
    # compute covariance
    covar = ma.cov(masked_boots, ddof=ddof, rowvar=rowvar)
    # set the diag. element to infinity and the off-diag elements to 0
    for i, is_masked in enumerate(np.diag(covar.mask)):
        if is_masked:
            covar.mask[i, :] = False
            covar.mask[:, i] = False
            covar[i, :] = 0.0
            covar[:, i] = 0.0
            covar[i, i] = np.inf
    return np.asarray(covar)


def write_fit_stats(fitparams, folder, precision=3, notation="decimal"):
    if not os.path.exists(folder):
        os.mkdir(folder)
    # write tex files
    with open(os.path.join(folder, "chi_squared.tex"), "w") as f:
        chisq = fitparams.chiSquare()
        string = format_variable(
            chisq, error=None, precision=precision, notation=notation)
        f.write("$\\chi^2 = %s$\n" % string.strip(" $"))
    with open(os.path.join(folder, "chi_squared_ndof.tex"), "w") as f:
        chisq /= fitparams.Ndof()
        string = format_variable(
            chisq, error=None, precision=precision, notation=notation)
        f.write("$\\chi^2_{\\rm dof} = %s$\n" % string.strip(" $"))


def write_parameters(fitparams, folder, precision=3, notation="auto"):
    # construct the header
    name_header = "# name"
    header = " ".join(fitparams.names)
    maxwidth = max(len(name) for name in fitparams.names)
    maxwidth = max(len(name_header), maxwidth)
    # create the output directory
    if not os.path.exists(folder):
        os.mkdir(folder)
    # write tex files for each parameter
    for name in fitparams.names:
        with open(os.path.join(folder, "%s.tex" % name), "w") as f:
            f.write("%s\n" % fitparams.paramAsTEX(
                name, precision=precision, notation=notation))
    # write a list of best fit parameters
    with open(os.path.join(folder, "parameters" + DEFAULT_EXT_DATA), "w") as f:
        f.write(
            "{:<{w}}    {:<16}    {:<16}\n".format(
                name_header, "value", "error", w=maxwidth))
        for name in fitparams.names:
            f.write(
                "{:>{w}}    {: 16.9e}    {: 16.9e}\n".format(
                    name, fitparams.paramBest(name),
                    fitparams.paramError(name), w=maxwidth))
    # write a list of bootstrap realisations
    np.savetxt(
        os.path.join(folder, "parameters" + DEFAULT_EXT_BOOT),
        fitparams.paramSamples(), header=header)
    # write the parameter covariance
    np.savetxt(
        os.path.join(folder, "parameters" + DEFAULT_EXT_COV),
        fitparams.paramCovar(), header=header)


def apply_bias(data, bias, renorm_bias=False):
    data_corrected = data.n / bias.n
    # compute the renormalisation to not alter the amplitude of the 
    norm_original = np.trapz(data.n, x=data.z)
    norm_corrected = np.trapz(data_corrected, x=data.z)
    renorm = norm_original / norm_corrected
    data_corrected *= renorm
    # apply also to the realisations
    real_corrected = data.getRealisations() / bias.getRealisations()
    norm_original = np.trapz(data.getRealisations(), x=data.z)
    norm_corrected = np.trapz(real_corrected, x=data.z)
    renorms = norm_original / norm_corrected
    real_corrected *= renorms[:, np.newaxis]
    # renormalize the bias data in place
    if renorm_bias:
        bias.n *= renorm
        bias.reals *= renorms[:, np.newaxis]
    # create a container with corrected redshift data
    container = RedshiftData(
        data.z, data_corrected, np.nanstd(real_corrected, axis=0))
    container.setRealisations(real_corrected)
    container.setCovariance(nancov(real_corrected))
    return container


def pack_model_redshifts(model, fitparams, z_list):
    # compute the best fit model realisations
    model_realisations = np.empty((
        len(z_list), len(fitparams), len(z_list[-1])))
    for i, params in enumerate(fitparams.paramSamples()):
        for j, n in enumerate(model.modelBest(params, z_list)[1]):
            model_realisations[j, i] = n
    # pack the results in containers
    model_containers = []
    for z, n in zip(*model.modelBest(fitparams, z_list)):
        model_containers.append(RedshiftData(z, n, np.zeros_like(z)))
    for i, realisations in enumerate(model_realisations):
        model_containers[i].dn = np.nanstd(realisations, axis=0)
        model_containers[i].setRealisations(realisations)
        model_containers[i].setCovariance(nancov(realisations))
    return model_containers


def write_nz_data(path, data, hdata=None, hboot=None, hcov=None, stats=False):
    assert(type(data) is RedshiftData)
    basepath = os.path.splitext(path)[0]
    print("writing n(z) data to: %s.*" % os.path.basename(basepath))
    if hdata is not None:
        nz = np.stack([data.z, data.n, data.dn]).T
        np.savetxt(basepath + DEFAULT_EXT_DATA, nz, header=hdata)
    if hboot is not None:
        np.savetxt(
            basepath + DEFAULT_EXT_BOOT, data.getRealisations(), header=hboot)
    if hcov is not None:
        np.savetxt(
            basepath + DEFAULT_EXT_COV, data.getCovariance(), header=hcov)
    # write mean and median redshifts
    if stats:
        statdir = os.path.join(os.path.dirname(path), "stats")
        if not os.path.exists(statdir):
            os.mkdir(statdir)
        # write tex files
        iterator = zip(
            ("mean", "median"), ("\\langle z \\rangle", "z_{\\rm med}"))
        for stat, TEX in iterator:
            try:
                zkey = "_" + binname(basepath)
            except ValueError:
                zkey = ""
            statfile = os.path.join(statdir, "%s%s.tex" % (stat, zkey))
            val = getattr(data, stat)()
            err = getattr(data, stat + "Error")()
            with open(statfile, "w") as f:
                string = format_variable(
                    val, error=err, TEX=True, precision=3, notation="decimal",
                    use_siunitx=True)
                f.write("$%s = %s$\n" % (TEX, string.strip("$")))


def write_global_cov(folder, data, order, header, prefix):
    assert(isinstance(folder, ScaleFolder))
    assert(type(data) is RedshiftDataBinned)
    print("writing global covariance matrix to: %s" % folder.basename())
    np.savetxt(
        folder.path_global_cov_file(prefix),
        data.getCovariance(), header=header)
    # store the order for later use
    with open(folder.path_bin_order_file(), "w") as f:
        for zbin in order:
            f.write("%s\n" % zbin)
