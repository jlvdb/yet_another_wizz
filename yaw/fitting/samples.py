from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import emcee
import emcee.autocorr
import numpy as np
import pandas as pd

from yaw.fitting.stats import Stats

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


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
        self.stats = Stats(self.samples, weights=None)
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
            n = int(3 * self.get_autocorr().max())
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
