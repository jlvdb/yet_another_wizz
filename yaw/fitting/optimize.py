from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import emcee
import emcee.autocorr
import numpy as np
from scipy.optimize import minimize

from yaw.core.math import apply_bool_mask_ndim, round_to

from yaw.fitting.samples import MCSamples

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from yaw.fitting.models import FitModel
    from yaw.fitting.priors import Prior


@dataclass
class Optimizer:

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

    def _log_prob_with_blobs(self, params: NDArray) -> tuple[float, float]:
        prior = self.log_prior(params)
        return self.log_like(params) + prior, prior

    def run_mcmc(
        self,
        *,
        p0: NDArray | None = None,
        nwalkers: int | None = None,
        max_steps: int = 10000,
        tau_scale: int = 50,
        tau_steps: int = 100,
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
            nwalkers, self.ndim, self._log_prob_with_blobs)

        # run the sampler, automatically expanding step number up to the limit
        end_message = "Sampling terminated on automatic convergence estimate."
        pbar = "sampling step {:d} of {:d} (est), max: {:d}\r"
        n_expect = 2 * tau_steps
        last_tau = sys.maxsize

        for _ in sampler.sample(p0, iterations=max_steps):
            it = sampler.iteration
            if it % 10 == 0:
                sys.stdout.write(pbar.format(it, n_expect, max_steps))

            # update autocorrelation time estimate
            if it % tau_steps == 0:
                tau = sampler.get_autocorr_time(tol=0).max()
                d_tau = np.abs(last_tau - tau) / tau
                n_expect = min(max_steps, round_to(tau_scale * tau, tau_steps))
                last_tau = tau
                if (tau * tau_scale < it) & (d_tau < tau_thresh):
                    break
        else:
            end_message = "Sampling terminated after reaching maximum steps."

        sys.stdout.write(
            f"sampling steps: {it:d} of {n_expect:d}: {end_message}\n")
        sys.stdout.flush()

        return MCSamples(sampler, self.parnames, self.neff)

    def run_fitting(self, p0: NDArray) -> tuple[NDArray, float]:
        opt = minimize(self.chi_squared, x0=p0, method="Nelder-Mead")
        status = "successful" if opt.success else "failed"
        sys.stdout.write(
            f"minimization {status} after {opt.nfev} evaluations: "
            f"{opt.message}\n")
        sys.stdout.flush()
        return opt.x, opt.fun  # best estimate, chi squared
