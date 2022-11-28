from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod
from typing import Iterator

import astropandas as apd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Interval, IntervalIndex

from yet_another_wizz.kdtree import EmptyKDTree, KDTree, SphericalKDTree


class Patch(ABC):

    patch_id = 0
    filepath = None
    _data = DataFrame()
    _has_z = False
    _has_weights = False

    def __len__(self) -> int:
        return len(self._data)

    def total(self) -> float:
        if self.has_weights():
            return self.weights.sum()
        else:
            return float(len(self))

    def is_loaded(self) -> bool:
        return self._data is not None

    def require_loaded(self) -> None:
        if not self.is_loaded():
            raise AttributeError("data is currently not loaded")

    @abstractmethod
    def load(self) -> None:
        NotImplemented

    @abstractmethod
    def unload(self) -> None:
        self._data = None

    def has_z(self) -> bool:
        return self._has_z

    def has_weights(self) -> bool:
        return self._has_weights

    @property
    def data(self) -> DataFrame:
        self.require_loaded()
        return self._data

    @property
    def ra(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["ra"].to_numpy()

    @property
    def dec(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["dec"].to_numpy()

    @property
    def z(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_z():
            return self._data["z"].to_numpy()
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_weights():
            return self._data["weights"].to_numpy()
        else:
            return None

    @abstractmethod
    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, Patch]]:
        # allow_no_redshift=True may be used for randoms and should yield self
        NotImplemented

    @abstractmethod
    def get_tree(self, **kwargs) -> KDTree:
        NotImplemented


class EmptyPatch(Patch):

    def __init__(
        self,
        patch_id: int,
        has_z: bool,
        has_weights: bool
    ) -> None:
        self.patch_id = patch_id
        self._has_z = has_z
        self._has_weights = has_weights
        # construct dummy dataframe with no data but the correct columns
        dummy_data = {
            "ra": pd.Series(dtype=np.float_),
            "dec": pd.Series(dtype=np.float_)}
        if has_z:
            dummy_data["z"] = pd.Series(dtype=np.float_)
        if has_weights:
            dummy_data["weights"] = pd.Series(dtype=np.float_)
        self._data = DataFrame(dummy_data)

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, EmptyPatch]]:
        if not allow_no_redshift and not self.has_z():
            raise ValueError("no redshifts for iteration provdided")
        for intv in IntervalIndex.from_breaks(z_bins):
            yield intv, self

    def get_tree(self, **kwargs) -> EmptyKDTree:
        return EmptyKDTree()


class PatchCatalog(Patch):

    def __init__(
        self,
        patch_id: int,
        data: DataFrame,
        filepath: str | None = None
    ) -> None:
        self.patch_id = patch_id
        if "ra" not in data:
            raise KeyError("right ascension column ('ra') is required")
        if "dec" not in data:
            raise KeyError("declination column ('dec') is required")
        if not set(data.columns) <= set(["ra", "dec", "z", "weights"]):
            raise KeyError(
                "'data' contains unidentified columns, optional columns are "
                "restricted to 'z' and 'weights'")
        # if there is a file path, store the file
        if filepath is not None:
            data.to_feather(filepath)
            self.filepath = filepath
        self._data = data
        self._has_z = "z" in data
        self._has_weights = "weights" in data

    def load(self) -> None:
        if not self._is_loaded:
            if self.filepath is None:
                raise ValueError("no datapath provided to load the data")
            self._data = pd.read_feather(self.filepath)

    def unload(self) -> None:
        self._data = None

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, PatchCatalog]]:
        if not allow_no_redshift and not self.has_z():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in IntervalIndex.from_breaks(z_bins):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(pd.cut(self.z, z_bins)):
                yield intv, PatchCatalog(self.patch_id, bin_data)

    def get_tree(self, **kwargs) -> SphericalKDTree:
        return SphericalKDTree(self.ra, self.dec, self.weights, **kwargs)


class PatchCollection:

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        patch_name: str,
        *,
        z_name: str | None = None,
        weight_name: str | None = None,
        n_patches: int | None = None,
        patch_directory: str | None = None
    ) -> None:
        if len(data) == 0:
            raise ValueError("data catalog is empty")
        # check if the columns exist
        renames = {ra_name: "ra", dec_name: "dec", patch_name: "patch"}
        if z_name is not None:
            renames[z_name] = "z"
        if weight_name is not None:
            renames[weight_name] = "weights"
        for col_name, kind in renames.items():
            if col_name not in data:
                raise KeyError(f"column {kind}='{col_name}' not found in data")
        # check if patches should be written and unloaded from memory
        unload = patch_directory is not None
        if unload:
            if not os.path.exists(patch_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{patch_directory}'")
        # run groupby first to avoid any intermediate copies of full data
        patches: dict[int, PatchCatalog] = {}
        for patch_id, patch_data in data.groupby(patch_name):
            if patch_id < 0:
                raise ValueError("negative patch IDs are not supported")
            patch = patch_data.rename(columns=renames).reset_index(drop=True)
            patch = patch.astype({"patch": np.int_})
            kwargs = {}
            if unload:
                # data will be written as feather file and loaded on demand
                kwargs["filepath"] = os.path.join(
                    patch_directory, f"patch_{patch_id:d}.feather")
            patch = PatchCatalog(
                int(patch_id), patch.drop("patch", axis=1), **kwargs)
            if unload:
                patch.unload()
                gc.collect()
            patches[int(patch_id)] = patch
        # assume that regions are labelled by integers, so fill up any empty
        # regions with dummies
        if n_patches is None:
            n_patches = max(patches.keys())
        ref_patch = next(iter(patches.values()))  # just any non-empty patch
        for patch_id in range(n_patches):
            if patch_id not in patches:
                patches[patch_id] = EmptyPatch(
                    patch_id, ref_patch.has_z(), ref_patch.has_weights())
        self.patches = patches

    @classmethod
    def from_file(
        cls,
        filepath: str,
        ra_name: str,
        dec_name: str,
        patch_name: str,
        *,
        z_name: str | None = None,
        weight_name: str | None = None,
        n_patches: int | None = None,
        patch_directory: str | None = None
    ) -> PatchCatalog:
        columns = [
            col for col in [ra_name, dec_name, patch_name, z_name, weight_name]
            if col is not None]
        data = apd.read_auto(filepath, columns=columns)
        return cls(
            data, ra_name, dec_name, patch_name,
            z_name=z_name,
            weight_name=weight_name,
            n_patches=n_patches,
            patch_directory=patch_directory)

    def __len__(self) -> int:
        return len(self.patches)

    def total(self) -> float:
        return np.sum([patch.total() for patch in self.patches.values()])

    def __iter__(self) -> Iterator[Patch]:
        for patch_id in sorted(self.patches.keys()):
            yield self.patches[patch_id]

    def is_loaded(self) -> list[bool]:
        return [patch.is_loaded() for patch in iter(self)]

    def load(self) -> None:
        return [patch.load() for patch in iter(self)]

    def unload(self) -> None:
        return [patch.unload() for patch in iter(self)]

    def has_z(self) -> bool:
        return all(patch.has_z() for patch in iter(self))  # always equal

    def has_weights(self) -> bool:
        return all(patch.has_weights() for patch in iter(self))  # always equal

    def get_z_min(self) -> float:
        return np.min([patch.z.min() for patch in iter(self)])

    def get_z_max(self) -> float:
        return np.max([patch.z.max() for patch in iter(self)])

    @property
    def ra(self) -> NDArray[np.float_]:
        return np.concatenate([patch.ra for patch in iter(self)])

    @property
    def dec(self) -> NDArray[np.float_]:
        return np.concatenate([patch.dec for patch in iter(self)])

    @property
    def z(self) -> NDArray[np.float_]:
        if self.has_z():
            return np.concatenate([patch.z for patch in iter(self)])
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        if self.has_weights():
            return np.concatenate([patch.weights for patch in iter(self)])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.float_]:
        return np.concatenate([
            np.full(len(patch), patch.patch_id) for patch in iter(self)])
