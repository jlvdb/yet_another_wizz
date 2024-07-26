from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Iterator, Sized
from itertools import repeat
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from yaw.catalog.patch import Patch, groupby_binning
from yaw.catalog.utils import Tclosed, logarithmic_mid
from yaw.coordinates import Coordinates, CoordsSky, DistsSky

__all__ = [
    "AngularTree",
    "BinnedTrees",
]

Tpath = Union[Path, str]


def parse_ang_limits(ang_min: NDArray, ang_max: NDArray) -> NDArray[np.float64]:
    ang_min = np.atleast_1d(ang_min).astype(np.float64)
    ang_max = np.atleast_1d(ang_max).astype(np.float64)
    if ang_min.ndim != 1 or ang_max.ndim != 1:
        raise ValueError("'ang_min' and 'ang_max' must be 1-dim")
    if len(ang_min) != len(ang_max):
        raise ValueError("length of 'ang_min' and 'ang_max' does not match")

    if np.any(ang_min >= ang_max):
        raise ValueError("'ang_min' < 'ang_max' not satisfied")
    ang_range = np.column_stack((ang_min, ang_max))
    if np.any(ang_range < 0.0) and np.any(ang_range > np.pi):
        raise ValueError("'ang_min' and 'ang_max' not in range [0.0, pi]")
    return ang_range


def get_ang_bins(
    ang_range: NDArray, weight_scale: float | None, weight_res: int
) -> NDArray:
    log_range = np.log10(ang_range)
    if weight_scale is not None:
        log_bins = np.linspace(log_range.min(), log_range.max(), weight_res)
        # ensure that all ang_min/max scales are included in the bins
        log_bins = np.concatenate(log_bins, log_range.flatten())
    else:
        log_bins = log_range.flatten()
    log_bins = np.sort(np.unique(log_bins))
    return 10.0**log_bins


def dispatch_counts(counts: NDArray, cumulative: bool) -> NDArray:
    if cumulative:
        return np.diff(counts)
    return counts[1:]  # discard counts within [0, ang_min)


def get_counts_for_limits(
    counts: NDArray, ang_bins: NDArray, ang_limits: NDArray
) -> NDArray:
    final_counts = np.empty(len(ang_limits), dtype=counts.dtype)
    for i, (ang_min, ang_max) in enumerate(ang_limits):
        idx_min = np.argmin(np.abs(ang_bins - ang_min))
        idx_max = np.argmin(np.abs(ang_bins - ang_max))
        final_counts[i] = counts[idx_min:idx_max].sum()
    return final_counts


class AngularTree(Sized):
    def __init__(
        self,
        coords: Coordinates,
        weights: NDArray | None = None,
        *,
        leafsize: int = 16,
    ) -> None:
        self.num_records = len(coords)
        if weights is None:
            self.weights = None
            self.total = float(self.num_records)
        elif len(weights) != coords:
            raise ValueError("shape of 'coords' and 'weights' does not match")
        else:
            self.weights = np.asarray(weights).astype(np.float64, copy=False)
            self.total = float(self.weights.sum())

        self.tree = KDTree(coords.to_3d(), leafsize=leafsize, copy_data=True)

    def __len__(self) -> int:
        return self.num_records

    def count(
        self,
        other: AngularTree,
        ang_min: NDArray,
        ang_max: NDArray,
        *,
        weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray[np.float64]:
        ang_limits = parse_ang_limits(ang_min, ang_max)
        ang_bins = get_ang_bins(ang_limits, weight_scale, weight_res)

        cumulative = len(ang_bins) < 8  # approx. turnover in processing speed
        try:
            counts = self.tree.count_neighbors(
                other.tree,
                r=DistsSky(ang_bins).to_3d(),
                weights=(self.weights, other.weights),
                cumulative=cumulative,
            ).astype(np.float64)
        except IndexError:
            counts = np.zeros_like(ang_bins)
        counts = dispatch_counts(counts, cumulative)

        if weight_scale is not None:
            ang_weights = logarithmic_mid(ang_bins) ** weight_scale
            counts *= ang_weights / ang_weights.sum()
        return get_counts_for_limits(counts, ang_bins, ang_limits)


def build_binned_trees(
    patch: Patch,
    binning: NDArray,
    closed: str,
    leafsize: int,
) -> tuple[AngularTree]:
    if not patch.has_redshifts():
        raise ValueError("patch has no 'redshifts' attached")

    trees = []
    for _, bin_data in groupby_binning(
        patch.redshifts,
        binning,
        closed=closed,
        coords=patch.coords,
        weights=patch.weights,
    ):
        bin_data["coords"] = CoordsSky(bin_data["coords"])
        tree = AngularTree(**bin_data, leafsize=leafsize)
        trees.append(tree)
    return tuple(trees)


class BinnedTrees(Iterable):
    def __init__(self, patch: Patch) -> None:
        self.cache_path = patch.cache_path
        if not self.binning_file.exists():
            raise FileNotFoundError(f"no trees found for patch at '{self.cache_path}'")

        with self.binning_file.open() as f:
            binning = json.load(f)
        if binning is None:
            self.binning = None
        else:
            self.binning = np.asarray(binning)

    @classmethod
    def build(
        cls,
        patch: Patch,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = "left",
        leafsize: int = 16,
        force: bool = False,
    ) -> BinnedTrees:
        new = cls.__new__(cls)
        new.cache_path = patch.cache_path

        if binning is not None:
            binning = np.asarray(binning, dtype=np.float64, copy=False)
        if not force and new.binning_file.exists():
            old = cls(patch)
            if old.binning_equal(binning):
                return old

        with new.trees_file.open(mode="wb") as f:
            if binning is not None:
                trees = build_binned_trees(patch, binning, closed, leafsize)
            else:
                trees = AngularTree(patch.coords, patch.weights, leafsize=leafsize)
            pickle.dump(trees, f)

        with new.binning_file.open(mode="w") as f:
            try:
                json.dump(binning.tolist(), f)
            except AttributeError:
                json.dump(binning, f)
        return cls(patch)

    @property
    def binning_file(self) -> Path:
        return self.cache_path / "binning.json"

    @property
    def trees_file(self) -> Path:
        return self.cache_path / "trees.pkl"

    def is_binned(self) -> bool:
        return self.binning is not None

    def binning_equal(self, binning: NDArray | None) -> bool:
        if self.binning is None and binning is None:
            return True
        elif np.array_equal(self.binning, binning):
            return True
        return False

    @property
    def trees(self) -> AngularTree | tuple[AngularTree]:
        with self.trees_file.open(mode="rb") as f:
            return pickle.load(f)

    def __iter__(self) -> Iterator[AngularTree]:
        if self.is_binned():
            yield from self.trees
        yield from repeat(self.trees)

    def count_binned(
        self,
        other: BinnedTrees,
        ang_min: NDArray,
        ang_max: NDArray,
        weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray[np.float64]:
        is_binned = (self.is_binned(), other.is_binned())
        if not any(is_binned):
            raise ValueError("at least one of the trees must be binned")
        elif all(is_binned) and not self.binning_equal(other.binning):
            raise ValueError("binning of trees does not match")

        binned_counts = []
        for tree_self, tree_other in zip(iter(self), iter(other)):
            counts = tree_self.count(
                tree_other,
                ang_min,
                ang_max,
                weight_scale=weight_scale,
                weight_res=weight_res,
            )
            binned_counts.append(counts)
        return np.transpose(binned_counts)
