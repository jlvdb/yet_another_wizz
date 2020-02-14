import copy
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from yaw_tools.folders import (DEFAULT_EXT_BOOT, DEFAULT_EXT_COV,
                               DEFAULT_EXT_DATA)
from yaw_tools.utils import nancov, write_nz_data

from Nz_Fitting import RedshiftData


class Pickle(object):

    # uninitialized members
    _z = None
    _DD = None
    _DR = None
    _RR = None
    _zs = None
    _z = None
    _n_ref = None
    _bin_width_factor = None
    _sampling_idx = None
    # members with default values
    _n_bootstraps = 1000
    _jackknife = False

    def __str__(self):
        return "<%s with %d bins and %d regions>" % (
            self.__class__.__name__, self.n_bins, self.n_regions)

    def __eq__(self, other):
        if type(self) != type(other):
            raise TypeError
        elif self.shape != other.shape:
            return False
        elif np.any(self._bin_width_factor != other._bin_width_factor):
            return False
        else:
            z_equal = np.all(self._z == other._z)
            DD_equal = np.all(self._DD == other._DD)
            DR_equal = np.all(self._DR == other._DR)
            RR_equal = np.all(self._RR == other._RR)
            return z_equal & DD_equal & DR_equal & RR_equal

    def __ne__(self, other):
        if type(self) != type(other):
            raise TypeError
        if self.shape != other.shape:
            return True
        elif np.any(self._bin_width_factor != other._bin_width_factor):
            return True
        z_inequal = np.all(self._z != other._z)
        DD_inequal = np.any(self._DD != other._DD)
        DR_inequal = np.any(self._DR != other._DR)
        RR_inequal = np.any(self._RR != other._RR)
        return z_inequal & DD_inequal & DR_inequal & RR_inequal

    def __copy__(self):
        inst = type(self).__new__(self.__class__)
        inst._DD = self._DD.copy()
        inst._DR = self._DR.copy()
        # RR is None in the new instance
        if self._RR is not None:
            inst._RR = self._RR.copy()
        return inst

    def __getattr__(self, attr):
        if attr == "shape":
            return self._DD.shape
        elif attr == "n_bins":
            return self._DD.shape[0]
        elif attr == "n_regions":
            return self._DD.shape[1]
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (
                    self.__class__.__name__, attr))

    def _from_pairs(self, DD, DR, RR=None, n_ref=None, zs=None):
        assert(DD.shape == DR.shape)
        inst = type(self).__new__(self.__class__)
        inst._DD = DD
        inst._DR = DR
        inst._RR = RR
        inst._update_z()
        return inst

    @staticmethod
    def _load_pickle(path):
        try:  # python 3
            with open(path, "rb") as f:
                return pickle.load(f)
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")
        except TypeError:  # python 2
            with open(path) as f:
                return pickle.load(f)

    def _update_z(self):
        if self._n_ref is not None and self._zs is not None:
            self._z = self._zs.sum(axis=1) / self._n_ref.sum(axis=1)

    @staticmethod
    def _correlation_estimator(DD, DR, RR=None):
        if RR is None:  # use Peebles-Davis estimator
            est = DD / DR - 1.0
        else:  # use Landy-Szalay estimator
            est = (DD - 2.0 * DR + RR) / RR
        est[np.isinf(est)] = np.nan  # we cannot handle infs properly
        return est

    def join(self, other):
        if type(self) != type(other):
            raise TypeError(
                "unsupported operand type(s) for +: '%s' and '%s'" % (
                    self.__class__.__name__, other.__class__.__name__))
        if self.n_bins != other.n_bins:
            raise ValueError(
                "Number of redshift bins do not match")
        DD = np.concatenate([self._DD, other._DD], axis=1)
        DR = np.concatenate([self._DR, other._DR], axis=1)
        kwargs = {}
        for attr in ("_RR", "_n_ref", "_zs"):
            if getattr(self, attr) is None or getattr(other, attr) is None:
                kwargs[attr.strip("_")] = None
            else:
                kwargs[attr.strip("_")] = np.concatenate([
                    getattr(self, attr), getattr(other, attr)], axis=1)
        return self._from_pairs(DD, DR, **kwargs)

    def ingest(self, other):
        if type(self) != type(other):
            raise TypeError(
                "unsupported operand type(s) for +: '%s' and '%s'" % (
                    self.__class__.__name__, other.__class__.__name__))
        if self.n_bins != other.n_bins:
            raise ValueError(
                "Number of redshift bins do not match")
        self._update_z()
        self._DD = np.concatenate([self._DD, other._DD], axis=1)
        self._DR = np.concatenate([self._DR, other._DR], axis=1)
        for attr in ("_RR", "_n_ref", "_zs"):
            if getattr(self, attr) is None or getattr(other, attr) is None:
                setattr(self, attr, None)
            else:
                setattr(self, attr, np.concatenate([
                    getattr(self, attr), getattr(other, attr)], axis=1))
        self._update_z()
        return self

    def set_sampling_method(self, method):
        methods = ("jackknife", "bootstrap")
        if method not in methods:
            raise ValueError(
                "method must be either of: " + ", ".join(methods))
        assert(method in ("jackknife", "bootstrap"))
        new_state = method == "jackknife"
        if new_state != self._jackknife:
            self._jackknife = new_state
            self.generate_sampling_idx()

    def generate_sampling_idx(self, n_bootstraps=None):
        if self._jackknife:
            reg_idx = np.arange(0, self.n_regions, dtype=np.int_)
            idx = np.empty(
                (self.n_regions, self.n_regions - 1), dtype=np.int_)
            for i in range(self.n_regions):
                idx[i] = np.roll(reg_idx, i)[:-1]
        else:  # bootstrap
            if n_bootstraps is not None:
                self._n_bootstraps = n_bootstraps
            idx = np.random.randint(
                0, self.n_regions, size=(self._n_bootstraps, self.n_regions),
                dtype=np.int_)
        self._sampling_idx = idx

    def get_sampling_idx(self):
        return self._sampling_idx.copy()

    def set_sampling_idx(self, idx):
        idx = np.asarray(idx)
        if self._jackknife:
            shape_expect = (self.n_regions, self.n_regions - 1,)
            if idx.shape != shape_expect:
                raise ValueError(
                    "expected jackknifing region indices of shape "
                    "%s, but got shape %s" % (shape_expect, idx.shape))
        else:  # bootstrap
            shape_expect = (idx.shape[0], self.n_regions)
            if idx.shape != shape_expect:
                raise ValueError(
                    "expected bootstrapping region indices of shape "
                    "%s, but got shape %s" % (shape_expect, idx.shape))
        self._sampling_idx = idx

    def get_samples(self):
        # resample the correlation estimate
        DD = self._DD[:, self._sampling_idx].sum(axis=2)
        DR = self._DR[:, self._sampling_idx].sum(axis=2)
        try:
            RR = self._RR[:, self._sampling_idx].sum(axis=2)
            amplitude_samples = self._correlation_estimator(DD, DR, RR)
        except TypeError:
            amplitude_samples = self._correlation_estimator(DD, DR)
        if self._bin_width_factor is not None:  # only autocorrelation
            amplitude_samples *= self._bin_width_factor[:, np.newaxis]
        return amplitude_samples

    def get_redshifts(self):
        try:
            return self._z.copy()
        except Exception:
            raise ValueError("redshifts not set")

    def get_DD(self):
        return self._DD.copy()

    def get_DR(self):
        return self._DR.copy()

    def get_RR(self):
        return self._RR.copy()

    def get_amplitudes(self):
        # compute the correlation estimator
        DD = self._DD.sum(axis=1)
        DR = self._DR.sum(axis=1)
        try:
            RR = self._RR.sum(axis=1)
            amplitudes = self._correlation_estimator(DD, DR, RR)
        except AttributeError:
            amplitudes = self._correlation_estimator(DD, DR)
        # correct the amplitudes for variable redshift bin width
        if self._bin_width_factor is not None:
            amplitudes *= self._bin_width_factor
        return amplitudes

    def get_errors(self, method=None):
        if method is not None:
            self.set_sampling_method(method)
        # resample the correlation estimate
        amplitude_samples = self.get_samples()
        # estimate the standard error from the samples
        if self._jackknife:
            amplitude_means = np.nanmean(amplitude_samples, axis=1)
            variance = (self.n_regions - 1) / self.n_regions * np.nansum(
                (amplitude_samples - amplitude_means[:, np.newaxis])**2,
                axis=1)
            errors = np.sqrt(variance)
        else:  # bootstrap
            errors = np.nanstd(amplitude_samples, axis=1)
        # correct the amplitudes for variable redshift bin width
        return errors

    def get_data(self):
        data = RedshiftData(
            self._z, self.get_amplitudes(), self.get_errors())
        return data

    def check_plot(self, z=None, dz=0.0, ax=None):
        if ax is None:
            ax = plt.gca()
        if z is None:
            if self._z is None:
                z = np.arange(self.n_bins)
            else:
                z = self.get_redshifts()
        ax.errorbar(
            z + dz, self.get_amplitudes(), yerr=self.get_errors(),
            marker=".", ls="none")
        ax.set_xlim(left=0.0)
        ymin = ax.get_ylim()[0]
        ax.set_ylim(bottom=min(ymin, 0.0))


class CrossCorrelationPickle(Pickle):

    def __init__(self, path):
        data_dict = self._load_pickle(path)
        # incorporate data
        self._DD = data_dict["unknown"]
        self._DR = data_dict["rand"]
        self._n_ref = data_dict["n_reference"]
        self._zs = data_dict["redshift"]
        self._update_z()
        # initialize resampling
        self.generate_sampling_idx()


class AutoCorrelationPickle(Pickle):

    def __init__(self, path):
        data_dict = self._load_pickle(path)
        # incorporate data
        self._DD = data_dict["unknown"]
        self._DR = data_dict["rand"]
        self._n_ref = data_dict["n_reference"]
        self._zs = data_dict["redshift"]
        self._bin_width_factor = data_dict["amplitude_factor"]
        self._update_z()
        # initialize resampling
        self.generate_sampling_idx()


def merge_pickles(pickle_list):
    pickles = []
    for path in pickle_list:
        with open(path, "rb") as f:
            pickles.append(pickle.load(f))
    # create the master pickle
    master_pickle = {"n_regions": 0}
    for key in pickles[0]:
        try:
            master_pickle[key] = np.concatenate(
                [data[key] for data in pickles], axis=1)
        except ValueError:
            if key == "n_regions":  # sum the region counters
                master_pickle[key] = sum(data[key] for data in pickles)
            else:  # just copy first from first element
                master_pickle[key] = pickles[0][key]
    return master_pickle


class PickleConverter(object):

    bias = None
    bias_samples = None

    def __init__(self, n_boot=1000, boot_idx=None):
        if boot_idx is None:
            self.n_boot = n_boot
        else:
            self.n_boot = len(boot_idx)
        self.boot_idx = boot_idx

    def set_bias(self, bias, bias_samples):
        self.bias = bias
        self.bias_samples = bias_samples

    def load_pickle(self, pickle_path, pickle_class):
        self.pickle_path = pickle_path
        self.pickler = pickle_class(pickle_path)
        # initialize the bootstrap resampling indices
        if self.boot_idx is None:
            self.pickler.generate_sampling_idx(n_bootstraps=self.n_boot)
            self.boot_idx = self.pickler.get_sampling_idx()
        else:
            self.pickler.set_sampling_idx(self.boot_idx)
        # create the realisations
        self.redshifts = self.pickler.get_redshifts()
        self.amplitudes = self.pickler.get_amplitudes()
        self.samples = self.pickler.get_samples().T
        if self.bias is None and self.bias_samples is None:
            self.errors = self.pickler.get_errors()
        else:
            self.amplitudes /= self.bias
            self.samples /= self.bias_samples
            # compute errors of bias corrected correlation amplitudes
            self.errors = np.nanstd(self.samples, axis=0)

    def get_amplitudes(self):
        return self.amplitudes.copy()

    def get_samples(self):
        return self.samples.copy()

    def get_all(self):
        data = RedshiftData(self.redshifts, self.amplitudes, self.errors)
        data.setRealisations(self.samples)
        try:
            data.setCovariance(nancov(self.samples))
        except ValueError:
            with open("/net/fohlen12/home/jlvdb/samples.pkl", "wb") as f:
                pickle.dump(self.samples, f)
            exit()
        return data

    def write_output(self, scaledir, header_key, stats=False):
        path = scaledir.incorporate(self.pickle_path, ".*")
        header = "col 1 = mean redshift\n"
        header += "col 2 = correlation amplitude (%s)\n" % header_key
        header += "col 3 = amplitude error"
        write_nz_data(
            scaledir.incorporate(self.pickle_path, ".*"), self.get_all(),
            hdata=header,
            hboot="correlation amplitude (%s) realisations" % header_key,
            hcov="correlation amplitude (%s) covariance matrix" % header_key,
            stats=stats)
