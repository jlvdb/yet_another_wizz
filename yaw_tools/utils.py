import argparse

import numpy as np
from numpy import ma


DEFAULT_CAT_EXT = 1
DEFAULT_REGION_NO = 7
DEFAULT_RESAMPLING = 10
DEFALUT_PAIR_WEIGHTING = "Y"


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
