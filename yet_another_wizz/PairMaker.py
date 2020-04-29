import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from .spatial import count_pairs
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

    def _packData(self, RA, DEC, Z=None, weights=None, region_idx=None):
        # compare the input data vector lengths
        data = pd.DataFrame({"RA": RA, "DEC": DEC})
        if Z is not None:
            data["z"] = Z
        if weights is not None:
            data["weights"] = weights
            data["weights"] /= weights.mean()
        # set region indices
        if region_idx is not None:
            data["region_idx"] = region_idx.astype(np.uint8)
            # remove objects with negative indices
            data = data[data.region_idx >= 0]
        else:
            data["region_idx"] = np.zeros(len(RA), dtype=np.uint8)
        return data

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

    @staticmethod
    def _inputGood(*args):
        n_data = None
        for arg in args:
            if arg is not None:
                if n_data is None:
                    n_data = len(arg)
                if len(arg) != n_data:
                    return False
        return n_data > 0  # otherwise this is also bad

    def getReference(self):
        if self._reference_data is None:
            self._throwException(
                "reference data not initialized", RuntimeError)
        return self._reference_data

    def setReference(self, RA, DEC, Z, weights=None, region_idx=None):
        if not self._inputGood(RA, DEC, Z, weights, region_idx):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("loading %d objects\n" % len(RA))
        self._reference_data = self._packData(RA, DEC, Z, weights, region_idx)
        self._printMessage(
            "kept %d of %d objects\n" % (len(self._reference_data), len(RA)))

    def writeReference(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getReference().to_parquet(path)

    def getUnknown(self):
        if self._unknown_data is None:
            self._throwException(
                "unknown data not initialized", RuntimeError)
        return self._unknown_data

    def setUnknown(self, RA, DEC, Z=None, weights=None, region_idx=None):
        if not self._inputGood(RA, DEC, Z, weights, region_idx):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("loading %d objects\n" % len(RA))
        self._unknown_data = self._packData(RA, DEC, Z, weights, region_idx)
        self._printMessage(
            "kept %d of %d objects\n" % (len(self._unknown_data), len(RA)))

    def writeUnknown(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getUnknown().to_parquet(path)

    def getRandoms(self):
        if self._random_data is None:
            self._throwException(
                "random data not initialized", RuntimeError)
        return self._random_data

    def setRandoms(self, RA, DEC, Z=None, weights=None, region_idx=None):
        if not self._inputGood(RA, DEC, Z, weights, region_idx):
            self._throwException(
                "input data empty or data length does not match", ValueError)
        self._printMessage("lading %d objects\n" % len(RA))
        self._random_data = self._packData(RA, DEC, Z, weights, region_idx)
        self._printMessage(
            "kept %d of %d objects\n" % (len(self._random_data), len(RA)))

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
        # create pool to find pairs DD
        poolDD = ThreadHelper(
            self.nRegions(), threads=min(self._threads, self.nRegions()))
        poolDD.add_iterable(self.getReference().groupby("region_idx"))
        if regionize_unknown:
            poolDD.add_iterable(self.getUnknown().groupby("region_idx"))
        else:
            poolDD.add_iterable([
                (r, self.getUnknown())
                for r in sorted(pd.unique(self.getReference().region_idx))])
        poolDD.add_constant((rmin, rmax))
        poolDD.add_constant(comoving)
        poolDD.add_constant(self.cosmology)
        poolDD.add_constant(inv_distance_weight)
        # create pool to find pairs DD
        poolDR = ThreadHelper(
            self.nRegions(), threads=min(self._threads, self.nRegions()))
        poolDR.add_iterable(self.getReference().groupby("region_idx"))
        if regionize_unknown:
            poolDR.add_iterable(self.getRandoms().groupby("region_idx"))
        else:
            poolDR.add_iterable([
                (r, self.getRandoms())
                for r in sorted(pd.unique(self.getReference().region_idx))])
        poolDR.add_constant((rmin, rmax))
        poolDR.add_constant(comoving)
        poolDR.add_constant(self.cosmology)
        poolDR.add_constant(inv_distance_weight)
        # set data to random ratio
        if D_R_ratio == "global":
            D_R_ratio = np.full(
                self.nRegions(),
                len(self.getUnknown()) / len(self.getRandoms()))
        elif D_R_ratio == "local":
            n_D = np.asarray([
                np.count_nonzero(self._unknown_data.region_idx == i)
                for i in range(self.nRegions())])
            n_R = np.asarray([
                np.count_nonzero(self._random_data.region_idx == i)
                for i in range(self.nRegions())])
            D_R_ratio = n_D / n_R
        else:
            try:
                assert(D_R_ratio > 0.0)
                D_R_ratio = np.full(self.nRegions(), D_R_ratio)
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
        # find DR pairs in regions running parallel threads
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
            ref_data = self._reference_data[["z", "region_idx", "weights"]]
        except KeyError:
            ref_data = self._reference_data[["z", "region_idx"]]
        self._pair_counts = pd.concat([ref_data, DD, DR], axis=1)

    def getDummyCounts(
            self, rmin, rmax, comoving=False, inv_distance_weight=True,
            reference_weights=False):
        self._scales = {"min": rmin, "max": rmax, "comoving": comoving}
        self._dist_weight = inv_distance_weight
        if reference_weights:
            return pd.DataFrame(
                columns=["z", "region_idx", "weights", "DD", "DR"])
        else:
            return pd.DataFrame(
                columns=["z", "region_idx", "DD", "DR"])

    def getCounts(self):
        if self._pair_counts is None:
            self.countPairs()
        return self._pair_counts

    def writeCounts(self, path):
        self._printMessage("writing data to parquet file:\n    %s\n" % path)
        self.getCounts().to_parquet(path)
