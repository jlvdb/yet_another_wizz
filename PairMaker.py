import os
import pickle
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from .spatial import FastSeparation2Angle, SphericalKDTree, count_pairs
from .utils import (BaseClass, ThreadHelper, generate_randoms,
                    measure_region_area, regionize_data)


class PairMaker(BaseClass):

    _threads = cpu_count()  # use all threads by default
    _unknown_data = None
    _random_data = None
    _reference_data = None
    _scales = None
    _dist_weight = True
    _pair_counts = None

    def __init__(
            self, map_path, n_regions, cosmology=None, threads=None,
            verbose=True):
        self._verbose = verbose
        self._printMessage("initializing correlator\n")
        # initialize the number of parallel threads
        if threads is not None:  # mast have 1 <= threads <= max_threads
            if type(threads) is not int and threads < 1:
                self._throwException(
                    "'threads' must be a positive integer", ValueError)
            self._threads = min(max(threads, 1), cpu_count())
        # initialize the stomp map and the jackknife regions
        self._printMessage("initializing stomp mask:\n    %s\n" % map_path)
        if not os.path.exists(map_path):
            self._throwException(
                "stomp mask not found: %s" % map_path, OSError)
        if type(n_regions) is not int and n_regions < 1:
            self._throwException(
                "'n_regions' must be a positive integer", ValueError)
        self.n_regions = n_regions
        self._mask_path = os.path.abspath(map_path)
        self._printMessage(
            "initializing %d jackknife regions\n" % self.n_regions)
        self._initializeMask()

    def _initializeMask(self):
        # measure the area of all regions
        pool = ThreadHelper(self.n_regions, threads=self._threads)
        pool.add_constant(self._mask_path)
        pool.add_constant(self.n_regions)
        pool.add_iterable(range(self.n_regions))
        region_areas = pool.map(measure_region_area)
        # normalize the areas to get the weight of the region
        region_areas = np.asarray(region_areas)
        self.region_weights = region_areas / region_areas.sum()

    def _regionizeData(self, RA, DEC, Z=None, weights=None):
        # compare the input data vector lengths
        data = pd.DataFrame({"RA": RA, "DEC": DEC})
        if Z is not None:
            data["z"] = Z
        if weights is not None:
            data["weights"] = weights
            data["weights"] /= weights.mean()
        # identify the stomp region each object belongs to in parallel threads
        pool = ThreadHelper(self._threads, threads=self._threads)
        pool.add_constant(self._mask_path)
        pool.add_constant(self.n_regions)
        pool.add_iterable(np.array_split(data, self._threads))
        region_idx = pool.map(regionize_data)
        # remove objects that lie outside the map
        data["stomp_region"] = np.concatenate(region_idx)
        return data[data.stomp_region != -1]

    def getMeta(self):
        meta_dict = {
            "cosmology": self.cosmology, "scale": self._scales,
            "inv_dist_weights": self._dist_weight, "n_regions": self.n_regions,
            "region_weights": self.region_weights}
        return meta_dict

    def writeMeta(self, path):
        self._printMessage("writing meta data to pickle:\n    %s\n" % path)
        pickle_path = os.path.splitext(path)[0] + ".pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.getMeta(), f)

    def _inputGood(*args):
        n_data = None
        for arg in args:
            if arg is not None:
                if n_data is None:
                    n_data = len(arg)
                if len(args) != n_data:
                    return False
        return n_data > 0  # otherwise this is also bad

    def getReference(self):
        if self._reference_data is None:
            self._throwException(
                "reference data not initialized", RuntimeError)
        return self._reference_data

    def setReference(self, RA, DEC, Z, weights=None):
        if not self._inputGood(RA, DEC, Z, weights):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("regionizing %d objects\n" % len(RA))
        self._reference_data = self._regionizeData(RA, DEC, Z, weights=weights)
        self._printMessage("kept %d objects\n" % len(self._reference_data))

    def writeReference(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getReference().to_parquet(path)

    def getUnknown(self):
        if self._unknown_data is None:
            self._throwException(
                "unknown data not initialized", RuntimeError)
        return self._unknown_data

    def setUnknown(self, RA, DEC, Z=None, weights=None):
        if not self._inputGood(RA, DEC, Z, weights):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("regionizing %d objects\n" % len(RA))
        self._unknown_data = self._regionizeData(RA, DEC, Z, weights=weights)
        self._printMessage("kept %d objects\n" % len(self._unknown_data))

    def writeUnknown(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getUnknown().to_parquet(path)

    def getRandoms(self):
        if self._random_data is None:
            self._throwException(
                "random data not initialized", RuntimeError)
        return self._random_data

    def setRandoms(self, RA, DEC, Z=None, weights=None):
        if not self._inputGood(RA, DEC, Z, weights):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("regionizing %d objects\n" % len(RA))
        self._random_data = self._regionizeData(RA, DEC, Z, weights=weights)
        self._printMessage("kept %d objects\n" % len(self._random_data))

    def generateRandoms(self, randoms_factor=10):
        unknown_data = self.getUnknown()
        # compute the number of randoms per region
        n_random = len(unknown_data) * randoms_factor
        randoms_per_thread = np.diff(
            np.linspace(0, n_random, self._threads + 1, dtype=int))
        self._printMessage("generating %d uniform random points\n" % n_random)
        # generate randoms in each region map in parallel threads
        pool = ThreadHelper(self._threads, threads=self._threads)
        pool.add_constant(self._mask_path)
        pool.add_iterable(randoms_per_thread)
        randoms_threads = pool.map(generate_randoms)
        # stack data from threads
        randoms = pd.concat(randoms_threads)
        # add redshifts by resampling values from the unknown data which will
        # allow redshift binning for auto-correlation measurements
        if "z" in unknown_data:
            random_z = np.random.choice(
                unknown_data.z, len(randoms), replace=True)
            self.setRandoms(randoms.RA, randoms.DEC, random_z)
        else:
            self.setRandoms(randoms.RA, randoms.DEC)

    def writeRandoms(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getRandoms().to_parquet(path)

    def countPairs(
            self, rmin, rmax, comoving=False, inv_distance_weight=True,
            D_R_ratio="global", regionize_unknown=True):
        if regionize_unknown and D_R_ratio == "local":
            D_R_ratio = "global"
        self._scales = {"min": rmin, "max": rmax, "comoving": comoving}
        self._dist_weight = inv_distance_weight
        # check if all data is present
        self.getReference()
        self.getUnknown()
        if self._random_data is None:
            self.generateRandoms()
        # create pool to find pairs DD
        poolDD = ThreadHelper(
            self.n_regions, threads=min(self._threads, self.n_regions))
        poolDD.add_iterable(self.getReference().groupby("stomp_region"))
        if regionize_unknown:
            poolDD.add_iterable(self.getUnknown().groupby("stomp_region"))
        else:
            poolDD.add_iterable([
                (r, self.getUnknown())
                for r in sorted(pd.unique(self.getReference().stomp_region))])
        poolDD.add_constant((rmin, rmax))
        poolDD.add_constant(comoving)
        poolDD.add_constant(self.cosmology)
        poolDD.add_constant(inv_distance_weight)
        # create pool to find pairs DD
        poolDR = ThreadHelper(
            self.n_regions, threads=min(self._threads, self.n_regions))
        poolDR.add_iterable(self.getReference().groupby("stomp_region"))
        if regionize_unknown:
            poolDR.add_iterable(self.getRandoms().groupby("stomp_region"))
        else:
            poolDR.add_iterable([
                (r, self.getRandoms())
                for r in sorted(pd.unique(self.getReference().stomp_region))])
        poolDR.add_constant((rmin, rmax))
        poolDR.add_constant(comoving)
        poolDR.add_constant(self.cosmology)
        poolDR.add_constant(inv_distance_weight)
        # set data to random ratio
        if D_R_ratio == "global":
            D_R_ratio = np.full(
                self.n_regions,
                len(self.getUnknown()) / len(self.getRandoms()))
        elif D_R_ratio == "local":
            n_D = np.asarray([
                np.count_nonzero(self._unknown_data.stomp_region == i)
                for i in range(self.n_regions)])
            n_R = np.asarray([
                np.count_nonzero(self._random_data.stomp_region == i)
                for i in range(self.n_regions)])
            D_R_ratio = n_D / n_R
        else:
            try:
                assert(D_R_ratio > 0.0)
                D_R_ratio = np.full(self.n_regions, D_R_ratio)
            except Exception:
                self._throwException(
                    "D_R_ratio must be either of 'local', 'global' or a "
                    "positive number", ValueError)
        # find DD pairs in regions running parallel threads
        self._printMessage("finding data-data pairs\n")
        try:
            DD = pd.concat(poolDD.map(count_pairs))
            DD.rename(columns={"pairs": "DD"}, inplace=True)
        except ValueError:
            DD = pd.DataFrame({"DD": []})
        self._printMessage("finding data-random pairs\n")
        try:
            DR = poolDR.map(count_pairs)
            for i, D_R in enumerate(D_R_ratio):
                DR[i].pairs *= D_R
            DR = pd.concat(DR)
            DR.rename(columns={"pairs": "DR"}, inplace=True)
        except Exception:
            DR = pd.DataFrame({"DR": []})
        # combine the pair counts with redshifts and region indices
        try:
            ref_data = self._reference_data[["z", "stomp_region", "weights"]]
        except KeyError:
            ref_data = self._reference_data[["z", "stomp_region"]]
        self._pair_counts = pd.concat([ref_data, DD, DR], axis=1)

    def getDummyCounts(
            self, rmin, rmax, comoving=False, inv_distance_weight=True,
            reference_weights=False):
        self._scales = {"min": rmin, "max": rmax, "comoving": comoving}
        self._dist_weight = inv_distance_weight
        if reference_weights:
            return pd.DataFrame(columns=["z", "stomp", "weights", "DD", "DR"])
        else:
            return pd.DataFrame(columns=["z", "stomp", "DD", "DR"])

    def getCounts(self):
        if self._pair_counts is None:
            self.countPairs()
        return self._pair_counts

    def writeCounts(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getCounts().to_parquet(path)
