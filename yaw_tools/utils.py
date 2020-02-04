import argparse

import numpy as np
from numpy import ma


DEFAULT_CAT_EXT = 1
DEFAULT_REGION_NO = 7
DEFAULT_RESAMPLING = 10
DEFALUT_PAIR_WEIGHTING = "Y"


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


def nancov(bootstraps):
    # mask infinite values
    mask = np.logical_not(np.isfinite(bootstraps))
    masked_boots = ma.array(bootstraps, mask=mask)
    # compute covariance
    covar = ma.cov(masked_boots, ddof=0)
    # find rows/columns with NaNs
    diag = np.diag(covar)
    idx = np.arange(len(diag))
    idx = idx[np.isnan(diag)]
    # set the diag. element to infinity and the off-diag elements to 0
    for i in idx:
        covar[i, :] = 0.0
        covar[:, i] = 0.0
        covar[i, i] = np.inf
    return covar
