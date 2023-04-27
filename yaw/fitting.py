from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, cycle, repeat
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from yaw.correlation import HistogramData, SampledValue
from yaw.utils import apply_bool_mask_ndim, cov_from_samples, rebin

if TYPE_CHECKING:
    from yaw.correlation import RedshiftData


_Tname = TypeVar("_Tname", bound=str)
_Tparam = TypeVar("_Tparam")


class FitResult(Generic[_Tname, _Tparam], Mapping[_Tname, _Tparam]):

    def __init__(self, ndof: int, chisq: float, **params: _Tparam) -> None:
        self._ndof = ndof
        self._chisq = chisq
        self._params = params

    def __repr__(self) -> str:
        string = self.__class__.__name__
        values = [f"chi^2/dof={self.chisq_red:.3f}"]
        for key, value in self._params.items():
            values.append(f"{key}={value}")
        string += f"({', '.join(values)})"
        return string

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, name: _Tname) -> _Tparam:
        return self._params[name]

    def __iter__(self) -> Iterator[_Tname]:
        return iter(self._params)

    def __contains__(self, name: _Tname) -> bool:
        return name in self._params

    @property
    def parnames(self) -> tuple[_Tname]:
        return tuple(self.keys())

    @property
    def ndof(self) -> int:
        return self._ndof

    @property
    def chisq(self) -> float:
        return self._chisq

    @property
    def chisq_red(self) -> float:
        return self.chisq / self.ndof

    def as_array(self) -> NDArray:
        return np.column_stack(self._params.values())


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
) -> FitResult[SampledValue]:
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

        names_A = [f"A{i}" for i in range(len(data))]
        names_dz = [f"dz{i}" for i in range(len(data))]
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

    samples = sampler.get_chain(discard=ndiscard, flat=True)
    ndof = np.count_nonzero(mask) - ndim
    popt, _ = curve_fit(
        fit_model.eval_curve_fit, xdata=None, ydata=data_masked,
        p0=np.median(samples, axis=0), sigma=cov_mat_masked)

    params = dict()
    for name, p, samp in zip(names_all, popt, samples.T):
        params[name] = SampledValue(p, samp, method="bootstrap")

    res = FitResult(
        ndof=ndof,
        chisq=chi_squared(popt, fit_model, data_masked, inv_mat_masked),
        **params)
    res.log_prob = sampler.get_log_prob(discard=ndiscard, flat=True)
    return res
