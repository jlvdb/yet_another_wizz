import os
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

from .utils import BaseClass


class PdfMaker(BaseClass):

    z_bins = None
    _zmin = None
    _zmax = None
    _region_pickle = None

    def __init__(self, pair_counts, autocorr, verbose=True):
        self._verbose = verbose
        # load the pair counts
        self._printMessage("loading pair counts\n")
        if type(pair_counts) is pd.DataFrame:
            self._pair_counts = pair_counts
        try:
            self._pair_counts = pd.read_parquet(pair_counts)
            assert(len(self._pair_counts) > 0)
        except AssertionError:
            self._throwException(
                "the pair count file is empty", ValueError)
        except Exception:
            self._throwException(
                "'pair_counts' must be either a valid parquet file or a "
                "pandas DataFrame", ValueError)
        self._zmin = self._pair_counts.z.min()
        self._zmax = self._pair_counts.z.max()
        self.n_regions = len(self._pair_counts.region_idx.unique())
        self.autocorr = autocorr

    @staticmethod
    def adaptiveBins(nbins, zmin, zmax, redshifts):
        z_sorted = np.sort(redshifts)
        idx = np.linspace(0, len(z_sorted) - 1, nbins + 1, dtype=int)
        return z_sorted[idx]

    @staticmethod
    def comovingBins(nbins, zmin, zmax, cosmology):
        cbinning = np.linspace(
            cosmology.comoving_distance(zmin).value,
            cosmology.comoving_distance(zmax).value, nbins + 1)
        redshift_array = np.linspace(0, 10.0, 5000)
        comov_array = cosmology.comoving_distance(
            redshift_array).value
        cosmo_spline = InterpolatedUnivariateSpline(
            comov_array, redshift_array)
        return cosmo_spline(cbinning)

    @staticmethod
    def linearBins(nbins, zmin, zmax):
        return np.linspace(zmin, zmax, nbins + 1)

    @staticmethod
    def logspaceBins(nbins, zmin, zmax):
        logbinning = np.linspace(
            np.log(1.0 + zmin), np.log(1.0 + zmax), nbins + 1)
        return np.exp(logbinning) - 1.0

    def getBinning(self):
        if self.z_bins is None:
            self._throwException("binning not set", RuntimeError)
        return self.z_bins

    def setBinning(self, bin_edges):
        if not np.all(np.diff(bin_edges) > 0.0):
            self._throwException("bins must have positive width", ValueError)
        self.z_bins = np.asarray(bin_edges)

    def generateBinning(
            self, nbins=25, zmin=None, zmax=None, bintype="comoving"):
        bin_types = ("adaptive", "comoving", "linear", "logspace")
        if bintype not in bin_types:
            self._throwException(
                "'bintype' must be either of {%s}" % ", ".join(bintype),
                ValueError)
        if zmin is not None:
            self._zmin = zmin
        if zmax is not None:
            self._zmax = zmax
        self._printMessage(
            "generating %d %s bins in range (%.2f, %.2f)\n" % (
                nbins, bintype, zmin, zmax))
        if bintype == "adaptive":
            self.z_bins = self.adaptiveBins(
                nbins, self._zmin, self._zmax, self._pair_counts.z)
        elif bintype == "comoving":
            self.z_bins = self.comovingBins(
                nbins, self._zmin, self._zmax, self.cosmology)
        elif bintype == "linear":
            self.z_bins = self.linearBins(nbins, self._zmin, self._zmax)
        else:
            self.z_bins = self.logspaceBins(nbins, self._zmin, self._zmax)

    def writeBinning(self, path):
        self._printMessage("writing redshift bins to:\n    %s\n" % path)
        np.savetxt(path, self.z_bins)

    def collapsePairCounts(self, pair_counts, zbins):
        self._printMessage("collapsing pair counts by bin and region\n")
        pair_counts_regionized = pair_counts.groupby(
            pair_counts.region_idx)
        n_bins = len(zbins) - 1
        # create output arrays
        n_ref = np.empty((n_bins, self.n_regions), dtype=np.int32)
        z_sum = np.empty((n_bins, self.n_regions))
        DD_sum = np.empty_like(z_sum)
        DR_sum = np.empty_like(z_sum)
        # collapse objects into region statistics
        for reg_idx, region in pair_counts_regionized:
            region_binned = region.groupby(
                pd.cut(region.z, zbins))
            for bin_idx, (zlimits, zbin) in enumerate(region_binned):
                n_ref[bin_idx, reg_idx] = len(zbin)
                z_sum[bin_idx, reg_idx] = zbin.z.sum()
                DD_sum[bin_idx, reg_idx] = zbin.DD.sum()
                DR_sum[bin_idx, reg_idx] = zbin.DR.sum()
        pickle_dict = {
            "n_reference": n_ref, "redshift": z_sum,
            "unknown": DD_sum, "rand": DR_sum, "n_regions": self.n_regions}
        if self.autocorr:
            pickle_dict["amplitude_factor"] = np.diff(zbins)
        self._region_pickle = pickle_dict

    def getRegionPickle(self):
        if self._region_pickle is None:
            self.collapsePairCounts(self._pair_counts, self.getBinning())
        return self._region_pickle

    def writeRegionPickle(self, path):
        region_pickle = self.getRegionPickle()
        self._printMessage("writing region pickle to:\n    %s\n" % path)
        with open(path, "wb") as f:
            pickle.dump(region_pickle, f)

    def getRedshifts(self):
        region_pickle = self.getRegionPickle()
        n_ref = region_pickle["n_reference"]
        redshifts = region_pickle["redshift"]
        return redshifts.sum(axis=1) / n_ref.sum(axis=1)

    def getAmplitudes(self, rescale=False):
        region_pickle = self.getRegionPickle()
        DD = region_pickle["unknown"]
        DR = region_pickle["rand"]
        amplitudes = DD.sum(axis=1) / DR.sum(axis=1) - 1.0
        if rescale:
            amplitudes *= region_pickle["amplitude_factor"]
        return amplitudes

    def getErrors(self, rescale=False, n_bootstraps=1000):
        region_pickle = self.getRegionPickle()
        DD = region_pickle["unknown"]
        DR = region_pickle["rand"]
        # generate bootstrap region indices
        boot_idx = np.random.randint(
            0, self.n_regions, size=(n_bootstraps, self.n_regions))
        # create bootstrapped pair counts
        samples_DD = DD[:, boot_idx].sum(axis=2)
        samples_DR = DR[:, boot_idx].sum(axis=2)
        samples = samples_DD / samples_DR - 1.0
        if rescale:
            samples *= self._region_pickle["amplitude_factor"][:, np.newaxis]
        # compute classical bootstrap error
        return np.nanstd(samples, axis=1)

    def getNz(self, rescale=False, n_bootstraps=1000):
        z = self.getRedshifts()
        amp = self.getAmplitudes(rescale=False)
        err = self.getErrors(rescale=False, n_bootstraps=n_bootstraps)
        return np.stack([z, amp, err]).T

    def writeNz(self, path):
        nz_array = np.stack([
            self.getRedshifts(), self.getAmplitudes(), self.getErrors()]).T
        self._printMessage(
            "writing redshift distribution to:\n    %s\n" % path)
        header = "col 1 = mean redshift\n"
        header += "col 2 = correlation amplitude\n"
        header += "col 3 = amplitude error"
        np.savetxt(path, nz_array, header=header)
