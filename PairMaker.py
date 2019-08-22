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
            "n_regions": self.n_regions, "region_weights": self.region_weights}
        return meta_dict

    def writeMeta(self, path):
        self._printMessage("writing meta data to pickle:\n    %s\n" % path)
        pickle_path = os.path.splitext(path)[0] + ".pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.getMeta(), f)

    def getReference(self):
        if self._reference_data is None:
            self._throwException(
                "reference data not initialized", RuntimeError)
        return self._reference_data

    def setReference(self, RA, DEC, Z, weights=None):
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

    def setRandoms(self, RA, DEC, weights=None):
        self._printMessage("regionizing %d objects\n" % len(RA))
        self._random_data = self._regionizeData(RA, DEC, weights=weights)
        self._printMessage("kept %d objects\n" % len(self._random_data))

    def generateRandoms(self, randoms_factor=10):
        # compute the number of randoms per region
        n_random = len(self._unknown_data) * randoms_factor
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
        self.setRandoms(randoms.RA, randoms.DEC)

    def writeRandoms(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getRandoms().to_parquet(path)

    def countPairs(self, rmin, rmax, comoving=False, global_density=True):
        self._scales = {"min": rmin, "max": rmax, "comoving": comoving}
        # check if all data is present
        self.getReference()
        self.getUnknown()
        if self._random_data is None:
            self.generateRandoms()
        # create pool to find pairs DD
        poolDD = ThreadHelper(
            self.n_regions, threads=min(self._threads, self.n_regions))
        poolDD.add_iterable(self.getReference().groupby("stomp_region"))
        poolDD.add_iterable(self.getUnknown().groupby("stomp_region"))
        poolDD.add_constant((rmin, rmax))
        poolDD.add_constant(comoving)
        poolDD.add_constant(self.cosmology)
        if global_density:
            poolDD.add_constant(len(self._unknown_data))
        # create pool to find pairs DD
        poolDR = ThreadHelper(
            self.n_regions, threads=min(self._threads, self.n_regions))
        poolDR.add_iterable(self.getReference().groupby("stomp_region"))
        poolDR.add_iterable(self.getRandoms().groupby("stomp_region"))
        poolDR.add_constant((rmin, rmax))
        poolDR.add_constant(comoving)
        poolDR.add_constant(self.cosmology)
        if global_density:
            poolDR.add_constant(len(self._random_data))
        # find DD pairs in regions running parallel threads
        self._printMessage("finding data-data pairs\n")
        DD = pd.concat(poolDD.map(count_pairs))
        DD.rename(columns={"pairs": "DD"}, inplace=True)
        self._printMessage("finding data-random pairs\n")
        DR = pd.concat(poolDR.map(count_pairs))
        DR.rename(columns={"pairs": "DR"}, inplace=True)
        # combine the pair counts with redshifts and region indices
        try:
            ref_data = self._reference_data[["z", "stomp_region", "weights"]]
        except KeyError:
            ref_data = self._reference_data[["z", "stomp_region"]]
        self._pair_counts = pd.concat([ref_data, DD, DR], axis=1)

    def getCounts(self):
        if self._pair_counts is None:
            self.countPairs()
        return self._pair_counts

    def writeCounts(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getCounts().to_parquet(path)
