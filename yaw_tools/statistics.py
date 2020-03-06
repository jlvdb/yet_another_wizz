import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def mean(z, n):
    p = n / np.trapz(n, x=z)
    return np.trapz(z * p, x=z)


def median(z, n):
    integrated = cumtrapz(n, x=z, initial=0.0)
    P = interp1d(integrated / integrated[-1], z)
    return P(0.5)


def KullbackLeibler(P, Q, x):
    assert(len(P) == len(x))
    assert(len(Q) == len(x))
    assert(np.all(P >= 0.0) & np.all(Q >= 0.0))
    D_KL = 0.0
    for p, q in zip(P / np.trapz(P, x), Q / np.trapz(Q, x)):
        if q > 0.0:
            D_KL += p * np.log(p / q)
        # else D_KL_i = 0.0
    return D_KL


def KolmogorovSmirnov(P, Q, x):
    assert(len(P) == len(x))
    assert(len(Q) == len(x))
    assert(np.all(P >= 0.0) & np.all(Q >= 0.0))
    P_cdf = cumtrapz(P, x, initial=0.0)
    Q_cdf = cumtrapz(Q, x, initial=0.0)
    D_KS = np.max(P_cdf / P_cdf[-1] - Q_cdf / Q_cdf[-1])
    return D_KS


def ChiSquare(P, Q, x):
    chisq = np.sum((P - Q)**2)
    return chisq
