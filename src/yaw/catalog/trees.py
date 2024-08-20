from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator, Sized
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from yaw.catalog.utils import groupby_binning
from yaw.containers import Tclosed, default_closed, parse_binning
from yaw.coordinates import AngularCoordinates, AngularDistances

if TYPE_CHECKING:
    from yaw.catalog.patch import Patch

__all__ = [
    "AngularTree",
    "BinnedTrees",
]


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

    return 10.0 ** np.sort(np.unique(log_bins))


def logarithmic_mid(edges: NDArray) -> NDArray:
    log_edges = np.log10(edges)
    log_mids = (log_edges[:-1] + log_edges[1:]) / 2.0
    return 10.0**log_mids


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
    __slots__ = ("num_records", "weights", "total", "tree")

    def __init__(
        self,
        coords: AngularCoordinates,
        weights: NDArray | None = None,
        *,
        leafsize: int = 16,
    ) -> None:
        self.num_records = len(coords)

        if weights is None:
            self.weights = None
            self.total = float(self.num_records)

        elif len(weights) != self.num_records:
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
                r=AngularDistances(ang_bins).to_3d(),
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
    extra_attrs = {} if patch.weights is None else {"weights": patch.weights}

    trees = []
    for _, bin_data in groupby_binning(
        patch.redshifts,
        binning,
        closed=closed,
        coords=patch.coords,
        **extra_attrs,
    ):
        bin_data["coords"] = AngularCoordinates(bin_data["coords"])
        tree = AngularTree(**bin_data, leafsize=leafsize)
        trees.append(tree)

    return tuple(trees)


class BinnedTrees(Iterable):
    __slots__ = ("_patch", "binning")

    def __init__(self, patch: Patch) -> None:
        self._patch = patch
        if not self.binning_file.exists():
            raise FileNotFoundError(f"no trees found for patch at '{self.cache_path}'")

        binning = np.fromfile(self.binning_file)
        self.binning = None if len(binning) == 0 else binning

    @classmethod
    def build(
        cls,
        patch: Patch,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = default_closed,
        leafsize: int = 16,
        force: bool = False,
    ) -> BinnedTrees:
        binning = parse_binning(binning, optional=True)

        try:
            assert not force
            new = cls(patch)  # trees exists, load the associated binning
            assert new.binning_equal(binning)
        except (AssertionError, FileNotFoundError):
            new = cls.__new__(cls)
            new._patch = patch
            new.binning = binning

            with new.trees_file.open(mode="wb") as f:
                if binning is None:
                    trees = AngularTree(patch.coords, patch.weights, leafsize=leafsize)
                else:
                    trees = build_binned_trees(patch, binning, closed, leafsize)
                pickle.dump(trees, f)

            if binning is None:
                binning = np.empty(0)  # zero bytes in binary representation
            binning.tofile(new.binning_file)

        return new

    def to_path(self) -> str:
        return str(self.cache_path)

    @property
    def cache_path(self) -> Path:
        return self._patch.cache_path

    @property
    def binning_file(self) -> Path:
        return self.cache_path / "binning"

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
        yield from (self.trees if self.is_binned() else repeat(self.trees))
