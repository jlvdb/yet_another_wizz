from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, Literal

import numpy as np
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike, NDArray

from yaw.meta import AsciiSerializable, BinwiseData, Indexer

Tcov_kind = Literal["full", "diag", "var"]
Tmethod = Literal["jackknife", "bootstrap"]
Tsampled = TypeVar("Tsampled", bound="SampledData")


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    method: Tmethod,
    rowvar: bool = False,
    kind: Tcov_kind = "full"
) -> NDArray:
    """Compute a joint covariance from a sequence of data samples.

    These samples can be jackknife or bootstrap samples (etc.). If more than one
    set of samples is provided, the samples are concatenated along the second
    axis (default) or along the first axis if ``rowvar=True``.

    Args:
        samples (:obj`:NDArray`, :obj:`Sequence[NDArray]`):
            One or many sets of data samples. The number of samples must be
            identical.
        method (:obj:`str`, optional):
            The resampling method that generated the samples, see
            :obj:`~yaw.config.options.Options.method`.
        rowvar (:obj:`bool`, optional):
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind (:obj:`str`, optional):
            Determines the kind of covariance computed, see
            :obj:`~yaw.config.options.Options.kind`.
    """
    if method not in Tmethod.__args__:
        raise ValueError(f"invalid sampling method '{method}'")
    if kind not in Tcov_kind.__args__:
        raise ValueError(f"invalid covariance kind '{kind}'")

    ax_samples = 1 if rowvar else 0
    ax_observ = 0 if rowvar else 1
    try:
        concat_samples = np.concatenate(samples, axis=ax_observ)
    except AxisError:
        concat_samples = samples

    num_samples = concat_samples.shape[ax_samples]
    num_observ = concat_samples.shape[ax_observ]
    if num_samples == 1:
        return np.full((num_observ, num_observ), np.nan)

    if method == "bootstrap":
        covmat = np.cov(concat_samples, rowvar=rowvar, ddof=1)
    elif method == "jackknife":
        covmat = np.cov(concat_samples, rowvar=rowvar, ddof=0) * (num_samples - 1)

    if kind == "diag":
        # get a matrix with only the main diagonal elements
        idx_diag = 0
        cov_diags = np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        try:
            for sample in samples:
                # go to next diagonal that contains correlations between samples
                idx_diag += sample.shape[ax_observ]
                # add just the diagonal values to the existing matrix
                cov_diags += np.diag(np.diag(covmat, k=-idx_diag), k=-idx_diag)
                cov_diags += np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        except IndexError:
            raise
        covmat = cov_diags

    elif kind == "var":
        covmat = np.diag(np.diag(covmat, k=0), k=0)

    return np.atleast_2d(covmat)


@dataclass(frozen=True)
class SampledData(BinwiseData):
    """TODO: better name?"""
    edges: NDArray
    data: NDArray
    samples: NDArray
    method: Tmethod = field(kw_only=True)
    covariance: NDArray = field(init=False)

    def __post_init__(self) -> None:
        if self.data.shape != (self.num_bins,):
            raise ValueError("unexpected shape of 'data' array")

        if self.samples.ndim != 2:
            raise ValueError("'samples' must be two-dimensional")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError("number of bins for 'data' and 'samples' do not match")

        if self.method not in Tmethod.__args__:
            raise ValueError(f"unknown sampling method '{self.method}'")

        covmat = cov_from_samples(self.samples, self.method)
        object.__setattr__(self, "covariance", covmat)

    @property
    def error(self) -> NDArray:
        return np.sqrt(np.diag(self.covariance))

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.samples.shape == other.samples.shape
            and self.method == other.method
            and np.all(self.data == other.data)
            and np.all(self.samples == other.samples)
            and np.all(self.binning == other.binning)
        )

    def __add__(self, other: object) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            edges=self.edges,
            data=self.data + other.data,
            samples=self.samples + other.samples,
            method=self.method,
        )

    def __sub__(self, other: object) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            edges=self.edges,
            data=self.data - other.data,
            samples=self.samples - other.samples,
            method=self.method,
        )

    def _make_slice(self, item: int | slice) -> SampledData:
        if not isinstance(item, (int, np.integer, slice)):
            raise TypeError("item selector must be a slice or integer type")

        cls = type(self)
        new = cls.__new__(cls)

        left = np.atleast_1d(self.left[item])
        right = np.atleast_1d(self.right[item])
        new.edges = np.append(left, right[-1])

        new.data = np.atleast_1d(self.data[item])

        new.samples = self.samples[:, item]
        if new.samples.ndim == 1:
            new.samples = np.atleast_2d(new.samples).T

        new.covariance = np.atleast_2d(self.covariance[item])[item]

        new.method = self.method
        return new

    def is_compatible(self, other: Any, require: bool = False) -> bool:
        if not super().is_compatible(other, require):
            return False

        if self.num_samples != other.num_samples:
            if not require:
                return False
            raise ValueError("number of samples do not agree")

        if self.method != other.method:
            if not require:
                return False
            raise ValueError("resampling method does not agree")

        return True

    def get_correlation(self) -> NDArray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdev = self.error
            corr = self.covariance / np.outer(stdev, stdev)

        corr[self.covariance == 0] = 0
        return corr


class CorrData(AsciiSerializable, SampledData):
    info: str
    def plot(): ...
    def plot_corr(): ...


class Shiftable(ABC):
    @abstractmethod
    def shift(): ...
    @abstractmethod
    def rebin(): ...


class RedshiftData(CorrData, Shiftable):
    @classmethod
    def from_corrfuncs(): ...
    @classmethod
    def from_corrdata(): ...


class HistData(CorrData, Shiftable):
    density: bool
