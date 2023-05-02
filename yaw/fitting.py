from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod, abstractproperty
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Union

import emcee
import emcee.autocorr
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from yaw.correlation import HistogramData
from yaw.stats import Stats
from yaw.utils import apply_bool_mask_ndim, cov_from_samples, rebin, round_to

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
        log_prior: NDArray = sampler.get_blobs()
        log_like = log_prob - log_prior

        nsteps, nchains, ndim = samples.shape
        if len(parnames) != ndim:
            raise ValueError(
                "length of 'parnames' does not match feature dimensions")
        index = pd.MultiIndex.from_product(
            [range(nsteps), range(nchains)], names=["step", "chain"])

        self.samples = pd.DataFrame(
            data=samples.reshape((nsteps * nchains, ndim)),
            index=index, columns=parnames)
        self.logprobs = pd.DataFrame(
            dict(
                log_like=log_like.flatten(),
                log_prior=log_prior.flatten(),
                log_prob=log_prob.flatten()),
            index=index)
        self._init(ndata)

    def _init(self, ndata: int) -> None:
        self.stats = Stats(self.samples, self.logprobs["log_like"])
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

    def get_logprobs(
        self,
        *,
        step: _Tslice = slice(None),
        chain: _Tslice = slice(None)
    ) -> pd.Series:
        return self.logprobs.loc(axis=0)[step, chain]

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
        new.logprobs = self.logprobs[n*self.nchains:]
        new._init(self.ndata)
        return new

    def set_best(self, best: Sequence, chisq: float) -> None:
        self._best = pd.Series(data=best, index=self.parnames, name="best")
        self._chisq = chisq

    def best(self) -> pd.Series:
        if self._best is None:
            idx = self.logprobs["log_like"].argmax()
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
            return -2 * self.logprobs["log_like"].max()
        else:
            return self._chisq

    def chisq_red(self) -> float:
        return self.chisq() / self.ndof


class Prior(ABC):

    @abstractmethod
    def __call__(self, value: float) -> float:
        NotImplemented

    @abstractmethod
    def draw_samples(
        self,
        n_draw: int,
        rng: np.random.Generator = None
    ) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()


@dataclass
class ImproperPrior(Prior):

    def __call__(self, value: float) -> float:
        return 0.0

    def draw_samples(
        self,
        n_draw: int,
        rng: np.random.Generator = None
    ) -> NDArray:
        raise NotImplementedError("cannot draw samples for an improper prior")


@dataclass
class UniformPrior(Prior):

    low: float
    high: float

    def __call__(self, value: float) -> float:
        if self.low <= value < self.high:
            return -np.log(self.high-self.low)
        else:
            return -np.inf

    def draw_samples(
        self,
        n_draw: int,
        rng: np.random.Generator = None
    ) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()
        return np.random.uniform(self.low, self.high, size=n_draw)


@dataclass
class GaussianPrior(Prior):

    mu: float
    sigma: float

    def __call__(self, value: float) -> float:
        return -0.5 * ((value - self.mu) / self.sigma)**2

    def draw_samples(
        self,
        n_draw: int,
        rng: np.random.Generator = None
    ) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()
        return np.random.normal(self.mu, self.sigma, size=n_draw)


class FitModel(ABC):

    @abstractproperty
    def ndim(self) -> int:
        NotImplemented

    @abstractproperty
    def parnames(self) -> list[str]:
        NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.parnames)})"

    @abstractmethod
    def __call__(self, params: NDArray) -> NDArray:
        NotImplemented


@dataclass(repr=False)
class ShiftModel(FitModel):

    binning: NDArray
    counts: NDArray
    target_bins: NDArray = field(default=None)

    def __post_init__(self) -> None:
        if self.target_bins is None:
            self.target_bins = self.binning

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parnames(self) -> list[str]:
        return ["log10_A", "dz"]

    def __call__(self, params: NDArray) -> NDArray:
        log_amp, shift = params
        bins = self.target_bins + shift
        values = rebin(
            bins_new=bins,
            bins_old=self.binning,
            counts_old=self.counts)
        return 10**log_amp * values


@dataclass(repr=False)
class ModelEnsemble(FitModel):

    models: Sequence[FitModel]

    @property
    def ndim(self) -> int:
        return sum(model.ndim for model in self.models)

    @property
    def parnames(self) -> list[str]:
        counter = Counter()
        parnames = []
        for model in self.models:
            pnames = model.parnames
            counter.update(pnames)
            parnames.extend(f"{pname}_{counter[pname]}" for pname in pnames)
        return parnames

    def __repr__(self) -> str:
        models = ", ".join(model.__class__.__name__ for model in self.models)
        return f"{self.__class__.__name__}({models})"

    def __call__(self, params: NDArray) -> NDArray:
        values = []
        ndim = 0
        for model in self.models:
            pars = params[ndim:ndim+model.ndim]
            values.append(model(pars))
            ndim += model.ndim
        if ndim != len(params):
            raise IndexError(f"expected {ndim} arguments, got {len(params)}")
        return np.concatenate(values)


@dataclass
class BayesianModel(ABC):

    data: NDArray
    inv_sigma: NDArray
    model: FitModel
    priors: list[Prior] | None = field(default=None, init=False)
    mask: NDArray[np.bool_] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.set_mask(None)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        model = self.model
        ndata = self.ndata
        ndof = self.ndof
        return f"{cls}({ndata=}, {ndof=}, {model=})"

    @property
    def ndata(self) -> int:
        return len(self.data)

    @property
    def neff(self) -> int:
        return np.count_nonzero(self.mask)

    @property
    def ndim(self) -> int:
        return self.model.ndim

    @property
    def ndof(self) -> int:
        return self.neff - self.ndim

    @property
    def parnames(self) -> list[str]:
        return self.model.parnames

    def set_mask(self, mask: NDArray[np.bool_] | None) -> None:
        if mask is None:
            mask = np.ones(len(self.data), dtype=np.bool_)
        elif len(mask) != len(self.data):
            raise IndexError("length of data and mask do not agree")
        self._data_masked = self.data[mask]
        self._inv_sigma_masked = apply_bool_mask_ndim(self.inv_sigma, mask)
        self.mask = mask

    def set_priors(self, priors: Sequence[Prior] | None) -> None:
        if priors is None:
            self.priors = None
        elif len(priors) != self.model.ndim:
            raise IndexError("number of priors does not match dimensions")
        else:
            self.priors = [p for p in priors]

    def chi_squared(self, params: NDArray) -> float:
        prediction = self.model(params)
        r = prediction[self.mask] - self._data_masked
        if self.inv_sigma.ndim == 2:
            chisq = r.T @ self._inv_sigma_masked @ r
        elif self.inv_sigma.ndim == 1:
            chisq = np.sum((r * self._inv_sigma_masked) ** 2)
        else:
            raise ValueError(
                f"cannot interpret inv_sigma with {self.inv_sigma.ndim} "
                "dimensions")
        return chisq

    def log_like(self, params: NDArray) -> float:
        return -0.5 * self.chi_squared(params)

    def log_prior(self, params: NDArray) -> float:
        if self.priors is None:
            return 0.0
        else:
            return sum(prior(par) for prior, par in zip(self.priors, params))

    def log_prob(self, params: NDArray) -> float:
        return self.log_like(params) + self.log_prior(params)

    def log_prob_with_prior(self, params: NDArray) -> tuple[float, float]:
        prior = self.log_prior(params)
        return self.log_like(params) + prior, prior

    def run_mcmc(
        self,
        *,
        p0: NDArray | None = None,
        nwalkers: int | None = None,
        max_steps: int = 10000,
        tau_scale: int = 50,
        tau_steps: int = 50,
        tau_thresh: float = 0.01
    ) -> MCSamples:
        if self.priors is None:
            raise ValueError("MCMC sampling requires to set parameter priors")
        if nwalkers is None:
            if p0 is not None:
                nwalkers = len(p0)
            else:
                nwalkers = 10 * self.ndim

        # generate the starting position
        if p0 is None:
            rng = np.random.default_rng()
            p0 = np.column_stack([
                prior.draw_samples(nwalkers, rng)
                for prior in self.priors])

        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, self.log_prob_with_prior)

        # run the sampler, automatically expanding step number up to the limit
        end_message = "Sampling terminated on automatic convergence estimate"
        pbar = "sampling steps: {:d} of {:d} (est), max: {:d}\r"
        n_expect = 2 * tau_steps
        last_tau = sys.maxsize

        for _ in sampler.sample(p0, iterations=max_steps):
            it = sampler.iteration
            sys.stderr.write(pbar.format(it, n_expect, max_steps))

            # update autocorrelation time estimate
            if it % tau_steps == 0:
                tau = sampler.get_autocorr_time(tol=0).max()
                d_tau = np.abs(last_tau - tau) / tau
                n_expect = min(max_steps, round_to(tau_scale * tau, tau_steps))
                last_tau = tau
                if (tau * tau_scale < it) & (d_tau < tau_thresh):
                    break
        else:
            end_message = "Sampling terminated after reaching maximum steps"

        sys.stderr.write(
            pbar.format(it, n_expect, max_steps) + "\n")
        sys.stderr.writable(
            f"sampling steps: {it:d} of {n_expect:d}: {end_message}")
        sys.stderr.flush()

        return MCSamples(sampler, self.parnames, self.neff)

    def run_fitting(self, p0: NDArray) -> tuple[NDArray, float]:
        opt = minimize(self.chi_squared, x0=p0, method="Nelder-Mead")
        status = "successful" if opt.success else "failed"
        sys.stderr.write(
            f"minimization {status} after {opt.nfev} evaluations: "
            f"{opt.message}\n")
        sys.stderr.flush()
        return opt.x, opt.fun  # best estimate, chi squared


def shift_fit(
    data: Sequence[RedshiftData],
    model: Sequence[HistogramData],
    *,
    covariance: str = "var",
    nwalkers: int = None,
    max_steps: int = 10000,
    optimise: bool = True
) -> MCSamples:
    # verify that data and model have matching binning etc.
    for i, (bin_data, bin_model) in enumerate(zip(data, model), 1):
        if not bin_data.is_compatible(bin_model):
            raise ValueError(
                "'data' and 'model' are not compatible in binning, spatial "
                f"patches, or resampling method for the {i}-th entry")

    # build the joint data vector
    data_all = np.concatenate([bin_data.data for bin_data in data])
    cov_mat_all = cov_from_samples(
        [bin_data.samples for bin_data in data],
        method=bin_data.method, kind=covariance)
    # mask bad values
    var_all = np.diag(cov_mat_all)
    mask = np.isfinite(data_all) & np.isfinite(var_all) & (var_all > 0.0)

    # build the models
    fit_models = [
        ShiftModel(bin_model.edges, bin_model.data, bin_data.edges)
        for bin_model, bin_data in zip(model, data)]
    fit_model = ModelEnsemble(fit_models)
    full_model = BayesianModel(
        data_all, np.linalg.inv(cov_mat_all), fit_model)
    full_model.set_mask(mask)

    # add the priors
    priors: list[Prior] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bin_data, bin_model in zip(data, model):
            rebinned = bin_model.rebin(bin_data.edges)
            norm = np.trapz(rebinned.data, x=bin_data.mids)
            prior_log_amp = GaussianPrior(-np.log10(norm), 0.5)
            prior_dz = GaussianPrior(0.0, 0.02)
            priors.extend([prior_log_amp, prior_dz])
    full_model.set_priors(priors)

    samples = full_model.run_mcmc(
        nwalkers=nwalkers, max_steps=max_steps)
    if optimise:
        best, chisq = full_model.run_fitting(samples.best().to_numpy())
        samples.set_best(best, chisq)
    return samples
