from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import emcee
import emcee.autocorr
import h5py
import numpy as np
import pandas as pd

from yaw.core.abc import HDFSerializable
from yaw.core.utils import TypePathStr

from yaw.fitting.stats import Stats

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


_Tslice = Union[int, Sequence[int | bool], slice]
_Terror = Union[np.number, tuple[np.number, np.number]]


class FitData(ABC):

    def __iter__(self) -> Iterator[str]:
        return iter(self.parnames)

    @abstractproperty
    def parnames(self) -> list[str]: raise NotImplementedError

    @abstractproperty
    def ndata(self) -> int: raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.parnames)

    @property
    def ndof(self) -> int:
        return self.ndata - self.ndim

    @abstractmethod
    def chisq(self) -> float: raise NotImplementedError

    def chisq_red(self) -> float:
        return self.chisq() / self.ndof


@dataclass
class ParameterValue:

    value: np.number
    error: _Terror
    kind: str | None = field(default=None)

    def __repr__(self) -> str:
        string = self.__class__.__name__
        value = self.value
        if self.is_symmetric():
            error = f"{self.error:.3g}"
        else:
            error = f"{self.error[1]:+.3g}/{self.error[0]:+.3g}"
        kind = self.kind
        return f"{string}({value=:.3g}, {error=}, {kind=})"

    def __str__(self) -> str:
        string = f"{self.value:+.3g}"
        if self.is_symmetric():
            string += f"+/-{self.error:.3g}"
        else:
            string += f"{self.error[1]:+.3g}/{self.error[0]:+.3g}"
        return string

    def is_symmetric(self) -> bool:
        return not isinstance(self.error, (Sequence, np.ndarray))


class FitResult(FitData):

    def __init__(
        self,
        values: Mapping[str, np.number],
        errors: Mapping[_Terror],
        *,
        ndata: int,
        chisq: float
    ) -> None:
        # build paramater value mapping
        self._params: dict[str, ParameterValue] = dict()
        for name, value in values.items():
            try:
                error = errors[name]
            except KeyError as e:
                raise KeyError(f"missing error for parameter '{name}'") from e
            self._params[name] = ParameterValue(value, error)
        # meta data
        self._ndata = ndata
        self._chisq = chisq

    def __repr__(self) -> str:
        string = self.__class__.__name__
        values = [f"chi^2/dof={self.chisq_red():.3f}"]
        for name in self.parnames:
            values.append(f"{name}={str(self[name])}")
        string += f"({', '.join(values)})"
        return string

    def __getitem__(self, name: str) -> ParameterValue:
        return self._params[name]

    @property
    def parnames(self) -> list[str]:
        return list(self._params.keys())

    @property
    def ndata(self) -> int:
        return self._ndata

    def chisq(self) -> float:
        return self._chisq

    def to_file(self, fpath: TypePathStr) -> None:
        DELIM = " "
        PREC = 6
        WIDTH = max(max(len(name) for name in self), 7)
        is_symmetric = all(self[name].is_symmetric() for name in self)
        # format the meta data
        string = "# Fit result\n# ==========\n# meta data:\n"
        string += f"#   chisq={self.chisq():.8f}\n#   ndata={self.ndata}\n"
        string += "# values:\n"
        # format the header
        pad = lambda s: f"{s:>{PREC+7}}"
        header = [f"{'param':>{WIDTH}}", pad("value")]
        if is_symmetric:
            header.append(pad("error"))
        else:
            header.extend([pad("low"), pad("high")])
        header = DELIM.join(f"{name:>{WIDTH}}" for name in header)
        string += f"# {header[2:]}\n"
        # write the content
        for name in self:
            value = self[name]
            name = f"{name:>{WIDTH}}"
            values = [f"{value.value:+.{PREC}e}"]
            if is_symmetric:
                values.append(f"{value.error:+.{PREC}e}")
            else:
                values.append(f"{value.error[0]:+.{PREC}e}")
                values.append(f"{value.error[1]:+.{PREC}e}")
            string += f"{name}{DELIM}{DELIM.join(values)}\n"
        # write output
        with open(fpath, "w") as f:
            f.write(string.strip())

    @classmethod
    def from_file(cls, fpath: TypePathStr) -> FitResult:
        with open(fpath) as f:
            lines = f.readlines()
        values = {}
        errors = {}
        for line in lines:
            line = line.strip()
            if "chisq" in line:
                chisq = float(line.split("=")[1])
            elif "ndata" in line:
                ndata = int(line.split("=")[1])
            # parse the values
            elif not line.startswith("#"):
                splitted = line.split()
                try:
                    name, val, std = splitted
                    err = float(std)
                except ValueError:
                    name, val, low, high = splitted
                    err = (float(low), float(high))
                values[name] = float(val)
                errors[name] = err
        return cls(values, errors, ndata=ndata, chisq=chisq)


class MCSamples(HDFSerializable, FitData):

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
        self._ndata = ndata

    @property
    def ndata(self) -> int:
        return self._ndata

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
        if self._best is not None:
            new.set_best(self._best, self._chisq)
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

    def values(self, statistic: str = "median") -> pd.Series:
        if statistic not in ("best", "mean", "median", "mode"):
            raise ValueError(f"invalid statistic '{statistic}'")
        if statistic == "best":
            return self.best()
        else:
            return getattr(self.stats, statistic)()

    def errors(self, statistic: str = "std", *, sigma: float = 1.0) -> pd.Series:
        if statistic not in ("quantile", "std"):
            raise ValueError(f"invalid statistic '{statistic}'")
        return getattr(self.stats, statistic)(sigma)

    def chisq(self) -> float:
        if self._chisq is None:
            return -2 * self.logprobs["log_like"].max()
        else:
            return self._chisq

    def get_result(
        self,
        val_stat: str = "median",
        err_stat: str = "std",
        *,
        sigma: float = 1.0
    ) -> FitResult:
        values = self.values(val_stat).to_dict()
        errors = self.errors(err_stat, sigma=sigma)
        if isinstance(errors, pd.DataFrame):
            errors = {
                name: (low_high["low"], low_high["high"])
                for name, low_high in errors.to_dict().items()}
        else:
            errors = errors.to_dict()
        return FitResult(values, errors, ndata=self.ndata, chisq=self.chisq())

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> MCSamples:
        new = cls.__new__(cls)
        # restore the shared index (step and chain ID)
        group = source["index"]
        names = ["step", "chain"]
        index = pd.MultiIndex.from_arrays(
            [group[name][:] for name in names], names=names)
        # restore the samples
        group = source["samples"]
        new.samples = pd.DataFrame(index=index)
        for col in group.keys():
            new.samples[col] = group[col][:]
        # restore the prior, likelihood and posterior probabilities
        group = source["logprobs"]
        new.logprobs = pd.DataFrame(index=index)
        for col in group.keys():
            new.logprobs[col] = group[col][:]
        # restore the optional external best fit value
        if "best" in source:
            group = source["best"]
            best = [group[name][()] for name in group.keys()]
            chisq = group.attrs["chisq"]
            new.set_best(best, chisq)
        new._init(source.attrs["ndata"])
        return new

    def to_hdf(self, dest: h5py.Group) -> None:
        _compression = dict(fletcher32=True, compression="gzip", shuffle=True)
        dest.attrs["ndata"] = self.ndata
        # store the shared index (step and chain ID)
        group = dest.create_group("index", track_order=True)
        index = self.samples.index.to_frame()
        for col in index.columns:
            group.create_dataset(col, data=index[col], **_compression)
        # store the samples
        group = dest.create_group("samples", track_order=True)
        for col in self.parnames:
            group.create_dataset(col, data=self.samples[col], **_compression)
        # store the prior, likelihood and posterior probabilities
        group = dest.create_group("logprobs", track_order=True)
        for col in self.logprobs.columns:
            group.create_dataset(col, data=self.logprobs[col], **_compression)
        # store the optional external best fit value
        if self._best is not None:
            group = dest.create_group("best", track_order=True)
            for key, val in self._best.items():
                group.create_dataset(key, data=val)
            group.attrs["chisq"] = self._chisq
