from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import erf

from yaw import stats
from yaw.correlation import HistogramData
from yaw.utils import apply_bool_mask_ndim, cov_from_samples, rebin

if TYPE_CHECKING:
    from yaw.correlation import RedshiftData


class MCSamples:

    def __init__(
        self,
        sampler: emcee.EnsembleSampler,
        parnames: Sequence[str],
        ndata: int
    ) -> MCSamples:
        samples: NDArray = sampler.get_chain()
        self.nsteps, self.nchains, self.ndim = samples.shape
        self.ndata = ndata
        if len(parnames) != self.ndim:
            raise ValueError(
                "length of 'parnames' does not match feature dimensions")

        index = pd.MultiIndex.from_product(
            [range(self.nsteps), range(self.nchains)],
            names=["step", "chain"])
        self.samples = pd.DataFrame(
            data=samples.reshape((self.nsteps * self.nchains, self.ndim)),
            index=index, columns=parnames)

        log_prob: NDArray = sampler.get_log_prob()
        self.logprob = pd.Series(
            data=log_prob.flatten(),
            index=index, name="log_prob")

    def __repr__(self) -> str:
        nsamples = f"{self.nsteps}x{self.nchains}"
        ndof = self.ndof
        chisq = self.chisq()
        return f"{self.__class__.__name__}({nsamples=}, {ndof=}, {chisq=:.3f})"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, name: str) -> pd.Series:
        return self.samples[name]

    @property
    def parnames(self) -> list[str]:
        return list(self.samples.columns)

    @property
    def ndof(self) -> int:
        return self.ndata - self.ndim

    def get(self, *, step=slice(None), chain=slice(None)) -> pd.DataFrame:
        return self.samples.loc(axis=0)[step, chain]

    def discard(self, n: int) -> MCSamples:
        if n >= self.nsteps:
            raise ValueError(f"'n' exceeds nsteps={self.nsteps}")
        new = self.__class__.__new__(self.__class__)
        new.nsteps = self.nsteps - n
        new.nchains = self.nchains
        new.ndim = self.ndim
        new.ndata = self.ndata
        new.samples = self.samples[n*self.nchains:]
        new.logprob = self.logprob[n*self.nchains:]
        return new

    def best(self) -> pd.Series:
        idx = self.logprob.argmax()
        return self.samples.iloc[idx]

    def mean(self) -> pd.Series:
        stat = lambda x: np.average(x, weights=self.logprob)
        return self.samples.apply(stat)

    def median(self) -> pd.Series:
        stat = lambda x: stats.weighted_median(x, self.logprob)
        return self.samples.apply(stat)

    def mode(self) -> pd.Series:
        stat = lambda x: stats.weighted_mode(x, self.logprob)
        return self.samples.apply(stat)

    def values(self, statistic="median") -> pd.Series:
        if statistic not in ("best", "mean", "median", "mode"):
            raise ValueError(f"invalid statistic '{statistic}'")
        return getattr(self, statistic)()

    def quantile(self, sigma: float = 1.0) -> pd.Series:
        p = erf(sigma / np.sqrt(2.0))
        qs = [0.5 - p/2, 0.5 + p/2]
        df = pd.DataFrame(columns=self.parnames)
        for key, q in zip(["low", "high"], qs):
            stat = lambda x: stats.weighted_quantile(x, q, weights=self.logprob)
            df.loc[key] = self.samples.apply(stat)
        return df - self.median()

    def std(self, sigma: float = 1.0) -> pd.Series:
        stat = lambda x: stats.weighted_std(x, weights=self.logprob)
        return self.samples.apply(stat) * sigma

    def errors(self, sigma: float = 1.0, statistic="std") -> pd.Series:
        if statistic not in ("quantile", "std"):
            raise ValueError(f"invalid statistic '{statistic}'")
        return getattr(self, statistic)(sigma)

    def chisq(self) -> float:
        return -2 * self.logprob.max()

    def chisq_red(self) -> float:
        return self.chisq() / self.ndof


@dataclass
class FitModel(ABC):

    @abstractmethod
    def eval(self, *params) -> NDArray:
        NotImplemented

    @abstractmethod
    def eval_curve_fit(self, xdata_dummy: Any, *params) -> NDArray:
        NotImplemented


@dataclass
class ShiftModel(FitModel):

    binning: NDArray
    counts: NDArray
    target_bins: NDArray | None = field(default=None)
    mask: NDArray[np.bool_] = field(init=False)

    def get_target_bins(self) -> NDArray:
        if self.target_bins is None:
            return self.binning
        else:
            return self.target_bins

    def set_mask(self, mask: NDArray[np.bool_] | None):
        if mask is None:
            mask = np.ones(len(self.counts), dtype=np.bool_)
        else:
            n_expect = len(self.get_target_bins()) - 1
            if len(mask) != n_expect:
                raise IndexError(
                    "length of 'mask' and does not match binning, "
                    f"got {len(mask)}, expected {n_expect}")
        self.mask = mask

    def eval(self, amplitude: float, shift: float) -> NDArray:
        bins = self.get_target_bins() + shift
        values = rebin(
            bins_new=bins,
            bins_old=self.binning,
            counts_old=self.counts)
        return amplitude * values[self.mask]

    def eval_curve_fit(
        self,
        xdata_dummy: Any,
        amplitude: float,
        shift: float
    ) -> NDArray:
        return self.eval(amplitude, shift)


@dataclass
class ShiftModelEnsemble(FitModel):

    binning: Sequence[NDArray]
    counts: Sequence[NDArray]
    target_bins: Sequence[NDArray] | None = field(default=None)
    mask: Sequence[NDArray[np.bool_]] = field(init=False)
    models: Sequence[FitModel] = field(init=False)

    def __post_init__(self) -> None:
        self.models: list[ShiftModel] = []
        bin_iter = zip(
            self.binning, self.counts,
            repeat(None) if self.target_bins is None else self.target_bins)
        for bins, count, tbins in bin_iter:
            self.models.append(ShiftModel(bins, count, target_bins=tbins))

    def set_masks(self, masks: Sequence[ NDArray[np.bool_] | None]):
        for mask, model in zip(masks, self.models):
            model.set_mask(mask)

    def eval(
        self,
        *alternating_amp_shift: float,
        join: bool = True
    ) -> NDArray | list[NDArray]:
        n_expect = 2 * len(self.models)
        if len(alternating_amp_shift) != n_expect:
            raise IndexError(
                f"expected {n_expect} parameters for {len(self.models)} bins, "
                f"got {len(alternating_amp_shift)}")
        results = []
        for model, amplitude, shift in zip(
            self.models,
            alternating_amp_shift[0::2],
            alternating_amp_shift[1::2]
        ):
            results.append(model.eval(amplitude, shift))
        if join:
            return np.concatenate(results)
        else:
            return results

    def eval_curve_fit(
        self,
        xdata_dummy: Any,
        *alternating_amp_shift: float
    ) -> NDArray:
        return self.eval(*alternating_amp_shift)


def chi_squared(
    params: Sequence[float],
    model: FitModel,
    data: NDArray, 
    inv_sigma: NDArray
) -> float:
    r = model.eval(*params) - data
    if inv_sigma.ndim == 2:
        chisq = r.T @ inv_sigma @ r
    elif inv_sigma.ndim == 1:
        chisq = np.sum((r * inv_sigma) ** 2)
    else:
        raise ValueError(
            f"cannot interpret covariance with {inv_sigma.ndim} dimensions")
    return chisq


def log_prob(
    params: Sequence[float],
    model: FitModel,
    data: NDArray, 
    inv_sigma: NDArray
) -> float:
    return -0.5 * chi_squared(params, model, data, inv_sigma)


def shift_fit(
    data: Sequence[RedshiftData],
    model: Sequence[HistogramData],
    covariance: str = "var",
    nwalkers: int = None,
    nsteps: int = 300
) -> MCSamples:
    # verify that data and model have matching binning etc.
    for i, (bin_data, bin_model) in enumerate(zip(data, model), 1):
        if not bin_data.is_compatible(bin_model):
            raise ValueError(
                "'data' and 'model' are not compatible in binning, spatial "
                f"patches, or resampling method for the {i}-th entry")

    # build the joint data vector
    cov_mat = cov_from_samples(
        [bin_data.samples for bin_data in data],
        method=bin_data.method, kind=covariance)

    errors = [bin_data.get_error().to_numpy() for bin_data in data]
    masks = [
        (np.isfinite(bin_data.data) & np.isfinite(error) & (error > 0.0))
        for error, bin_data in zip(errors, data)]
    mask = np.concatenate(masks)

    data_masked = np.concatenate([
        bin_data.data[mask] for bin_data, mask in zip(data, masks)])
    cov_mat_masked = apply_bool_mask_ndim(cov_mat, np.concatenate(masks))
    inv_mat_masked = np.linalg.inv(cov_mat_masked)

    fit_model = ShiftModelEnsemble(
        binning=[bin_model.edges for bin_model in model],
        counts=[bin_model.data for bin_model in model],
        target_bins=[bin_data.edges for bin_data in data])
    fit_model.set_masks(masks)

    # run fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        guess_A = [
            1 / np.trapz(bin_model.rebin(bin_data.edges).data, x=bin_model.mids)
            for bin_data, bin_model in zip(data, model)]
        guess_dz = repeat(0.0)
        guess_all = list(chain.from_iterable(zip(guess_A, guess_dz)))

        width_A = (0.3 * A for A in guess_A)
        width_dz = repeat(0.05)
        width_all = list(chain.from_iterable(zip(width_A, width_dz)))

        names_A = [f"A{i+1}" for i in range(len(data))]
        names_dz = [f"dz{i+1}" for i in range(len(data))]
        names_all = list(chain.from_iterable(zip(names_A, names_dz)))

        ndim = len(guess_all)
        if nwalkers is None:
            nwalkers = 10 * ndim
        p0 = np.random.normal(guess_all, width_all, size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob,
            args=(fit_model, data_masked, inv_mat_masked))
        ndiscard = 100
        for _ in sampler.sample(p0, iterations=ndiscard+nsteps, progress=True):
            pass

    return MCSamples(sampler, names_all, len(data_masked))
