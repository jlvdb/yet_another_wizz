from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import TYPE_CHECKING, Any, Union

import emcee
import emcee.autocorr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from yaw.correlation import HistogramData
from yaw.stats import Stats
from yaw.utils import apply_bool_mask_ndim, cov_from_samples, rebin

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from yaw.correlation import RedshiftData


_Tslice = Union[int, Sequence[int | bool], slice]

class MCSamples:

    _best: pd.Series | None = None
    _chisq: float | None = None

    def __init__(
        self,
        sampler: emcee.EnsembleSampler,
        parnames: Sequence[str],
        ndata: int
    ) -> MCSamples:
        samples: NDArray = sampler.get_chain()
        log_prob: NDArray = sampler.get_log_prob()

        nsteps, nchains, ndim = samples.shape
        if len(parnames) != ndim:
            raise ValueError(
                "length of 'parnames' does not match feature dimensions")
        index = pd.MultiIndex.from_product(
            [range(nsteps), range(nchains)], names=["step", "chain"])

        self.samples = pd.DataFrame(
            data=samples.reshape((nsteps * nchains, ndim)),
            index=index, columns=parnames)
        self.logprob = pd.Series(
            data=log_prob.flatten(),
            index=index, name="log_prob")
        self._init(ndata)

    def _init(self, ndata: int) -> None:
        self.stats = Stats(self.samples, self.logprob)
        self.ndata = ndata

    @property
    def ndim(self) -> int:
        return len(self.parnames)

    @property
    def nchains(self) -> int:
        _, chains = self.samples.index.codes
        return len(np.unique(chains))

    @property
    def nsteps(self) -> int:
        steps, _ = self.samples.index.codes
        return len(np.unique(steps))

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

    def get_samples(
        self,
        *,
        step: _Tslice = slice(None),
        chain: _Tslice = slice(None)
    ) -> pd.DataFrame:
        return self.samples.loc(axis=0)[step, chain]

    def get_logprob(
        self,
        *,
        step: _Tslice = slice(None),
        chain: _Tslice = slice(None)
    ) -> pd.Series:
        return self.logprob.loc(axis=0)[step, chain]

    def get_autocorr(self) -> pd.Series:
        samples = self.samples.to_numpy()
        return pd.Series(
            emcee.autocorr.integrated_time(
                samples.reshape((self.nsteps, self.nchains, self.ndim)),
                quiet=True),
            index=self.parnames, name="tau")

    def discard(self, n: int | None = None) -> MCSamples:
        if n is None:
            n = int(2 * self.get_autocorr().max())
        if n >= self.nsteps:
            raise ValueError(f"'{n=}' exceeds nsteps={self.nsteps}")
        new = self.__class__.__new__(self.__class__)
        new.samples = self.samples[n*self.nchains:]
        new.logprob = self.logprob[n*self.nchains:]
        new._init(self.ndata)
        return new

    def set_best(self, best: Sequence, chisq: float) -> None:
        self._best = pd.Series(data=best, index=self.parnames, name="best")
        self._chisq = chisq

    def best(self) -> pd.Series:
        if self._best is None:
            idx = self.logprob.argmax()
            return self.samples.iloc[idx]
        else:
            return self._best

    def values(self, statistic="median") -> pd.Series:
        if statistic not in ("best", "mean", "median", "mode"):
            raise ValueError(f"invalid statistic '{statistic}'")
        if statistic == "best":
            return self.best()
        else:
            return getattr(self.stats, statistic)()

    def errors(self, sigma: float = 1.0, statistic="std") -> pd.Series:
        if statistic not in ("quantile", "std"):
            raise ValueError(f"invalid statistic '{statistic}'")
        return getattr(self.stats, statistic)(sigma)

    def chisq(self) -> float:
        if self._chisq is None:
            return -2 * self.logprob.max()
        else:
            return self._chisq

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
        return 10**amplitude * values[self.mask]

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


class Posterior:

    def __init__(self, mus: NDArray, sigmas: NDArray) -> None:
        self.mus = mus
        self.sigmas = sigmas

    def log_like(
        self,
        params: Sequence[float],
        model: FitModel,
        data: NDArray, 
        inv_sigma: NDArray
    ) -> float:
        return -0.5 * chi_squared(params, model, data, inv_sigma)

    def log_prior(self, params: Sequence[float]) -> float:
        priors = -0.5 * ((np.asarray(params) - self.mus) / self.sigmas)**2
        return priors.sum()

    def log_post(self, params: Sequence[float], *like_args) -> float:
        return self.log_like(params, *like_args) + self.log_prior(params)


def shift_fit(
    data: Sequence[RedshiftData],
    model: Sequence[HistogramData],
    covariance: str = "var",
    nwalkers: int = None,
    nsteps: int = 300,
    optimise: bool = True
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
    cov_mat_masked = apply_bool_mask_ndim(cov_mat, mask)
    inv_mat_masked = np.linalg.inv(cov_mat_masked)

    fit_model = ShiftModelEnsemble(
        binning=[bin_model.edges for bin_model in model],
        counts=[bin_model.data for bin_model in model],
        target_bins=[bin_data.edges for bin_data in data])
    fit_model.set_masks(masks)

    # run fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        guess_A = np.log10([
            1 / np.trapz(bin_model.rebin(bin_data.edges).data, x=bin_model.mids)
            for bin_data, bin_model in zip(data, model)])
        guess_dz = repeat(0.0)
        guess_all = list(chain.from_iterable(zip(guess_A, guess_dz)))

        width_A = (0.5 for _ in guess_A)
        width_dz = (0.02 for _ in guess_dz)
        width_all = list(chain.from_iterable(zip(width_A, width_dz)))

        names_A = [f"A{i+1}" for i in range(len(data))]
        names_dz = [f"dz{i+1}" for i in range(len(data))]
        names_all = list(chain.from_iterable(zip(names_A, names_dz)))

        cost = Posterior(guess_all, width_all)

        ndim = len(guess_all)
        if nwalkers is None:
            nwalkers = 10 * ndim
        p0 = np.random.normal(guess_all, width_all, size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, cost.log_post,
            args=(fit_model, data_masked, inv_mat_masked))
        for _ in sampler.sample(p0, iterations=nsteps, progress=True):
            pass

    samples = MCSamples(sampler, names_all, len(data_masked))
    if optimise:
        popt, _ = curve_fit(
            fit_model.eval_curve_fit,
            xdata=None,  # not needed
            ydata=data_masked,
            p0=samples.best().to_numpy(),
            sigma=cov_mat_masked)
        chisq = chi_squared(
            popt, fit_model, data_masked, inv_sigma=inv_mat_masked)
        samples.set_best(popt, chisq)
    return samples
