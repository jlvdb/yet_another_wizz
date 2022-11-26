import os
import struct
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from .spatial import count_pairs_binned
from .utils import BaseClass, ThreadHelper, dump_json


class PairMaker(BaseClass):

    _threads = cpu_count()  # use all threads by default
    _unknown_data = None
    _random_data = None
    _reference_data = None
    _scales = None
    _dist_weight = True
    _pair_counts = None

    def __init__(self, threads=None, verbose=True):
        self._verbose = verbose
        self._printMessage("initializing correlator\n")
        # initialize the number of parallel threads
        if threads is not None:  # mast have 1 <= threads <= max_threads
            if type(threads) is not int and threads < 1:
                self._throwException(
                    "'threads' must be a positive integer", ValueError)
            self._threads = min(max(threads, 1), cpu_count())
        self.setCosmology(name="default")

    def nRegions(self):
        if self._random_data is not None:
            return len(np.unique(self._random_data.region_idx))
        if self._unknown_data is not None:
            return len(np.unique(self._unknown_data.region_idx))
        if self._reference_data is not None:
            return len(np.unique(self._reference_data.region_idx))
        else:
            return 1

    def getMeta(self):
        meta_dict = {
            "cosmology": self._cosmo_info,
            "scale": self._scales,
            "inv_dist_weights": self._dist_weight,
            "n_regions": self.nRegions()}
        return meta_dict

    def writeMeta(self, path):
        json_path = os.path.splitext(path)[0] + ".json"
        self._printMessage("writing meta data to:\n    %s\n" % json_path)
        dump_json(self.getMeta(), json_path)

    def getReference(self):
        if self._reference_data is None:
            self._throwException(
                "reference data not initialized", RuntimeError)
        return self._reference_data

    def setReference(self, df):
        self._reference_data = df
        self._printMessage(f"loaded {len(df)} objects\n")

    def writeReference(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getReference().to_parquet(path)

    def getUnknown(self):
        if self._unknown_data is None:
            self._throwException(
                "unknown data not initialized", RuntimeError)
        return self._unknown_data

    def setUnknown(self, df):
        self._unknown_data = df
        self._printMessage(f"loaded {len(df)} objects\n")

    def writeUnknown(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getUnknown().to_parquet(path)

    def getRandoms(self):
        if self._random_data is None:
            self._throwException(
                "random data not initialized", RuntimeError)
        return self._random_data

    def setRandoms(self, df):
        self._random_data = df
        self._printMessage(f"loaded {len(df)} objects\n")

    def writeRandoms(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getRandoms().to_parquet(path)

    def countPairs(
            self, scales, zbins, inv_distance_weight=True,
            regionize_unknown=True):
        self._dist_weight = inv_distance_weight
        # check if all data is present
        self.getReference()
        self.getUnknown()
        self.getRandoms()
        # check that all the region indices match
        reference_regions = np.unique(self.getReference().region_idx)
        unknown_regions = np.unique(self.getUnknown().region_idx)
        if set(reference_regions) != set(unknown_regions):
            self._throwException(
                "region indices of unknown objects do not match reference",
                ValueError)
        randoms_regions = np.unique(self.getRandoms().region_idx)
        if set(reference_regions) != set(randoms_regions):
            self._throwException(
                "region indices of randoms do not match reference objects",
                ValueError)
        self._printMessage("regionising data sets\n")
        # create pool to find pairs DD
        poolDD = ThreadHelper(
            self.nRegions(), threads=min(self._threads, self.nRegions()))
        poolDD.add_constant(len(self.getReference()) == len(self.getUnknown()))  # auto
        reg_ids, regions = zip(*self.getReference().groupby("region_idx"))
        poolDD.add_iterable(reg_ids)
        poolDD.add_iterable(regions)
        regions = {}
        for reg_id, region in self.getUnknown().groupby("region_idx"):
            regions[reg_id] = region
        poolDD.add_iterable([{reg_id: regions[reg_id]} for reg_id in reg_ids])
        poolDD.add_constant(scales)
        poolDD.add_constant(self.cosmology)
        poolDD.add_constant(zbins)
        poolDD.add_constant(True)
        poolDD.add_constant(False)
        poolDD.add_constant(-1.0 if inv_distance_weight else None)
        poolDD.add_constant(50)
        # create pool to find pairs DR
        poolDR = ThreadHelper(
            self.nRegions(), threads=min(self._threads, self.nRegions()))
        poolDR.add_constant(False)  # auto
        reg_ids, regions = zip(*self.getReference().groupby("region_idx"))
        poolDR.add_iterable(reg_ids)
        poolDR.add_iterable(regions)
        regions = {}
        for reg_id, region in self.getRandoms().groupby("region_idx"):
            regions[reg_id] = region
        poolDR.add_iterable([{reg_id: regions[reg_id]} for reg_id in reg_ids])
        poolDR.add_constant(scales)
        poolDR.add_constant(self.cosmology)
        poolDR.add_constant(zbins)
        poolDR.add_constant(True)
        poolDR.add_constant(False)
        poolDR.add_constant(-1.0 if inv_distance_weight else None)
        poolDR.add_constant(50)
        # find DD pairs in regions running parallel threads
        self._printMessage("finding data-data pairs\n")
        DD = poolDD.map(count_pairs_binned)
        # find DR pairs in regions running parallel threads
        self._printMessage("finding data-random pairs\n")
        DR = poolDR.map(count_pairs_binned)
        self._pair_counts = (DD, DR)

    def getDummyCounts(
            self, rmin, rmax, comoving=False, inv_distance_weight=True,
            reference_weights=False):
        return (None, None)

    def getCounts(self):
        if self._pair_counts is None:
            self.countPairs()
        return self._pair_counts

    def writeCounts(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getCounts().to_parquet(path)
