from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from yaw.utils import apply_bool_mask_ndim, shift_histogram
from yaw.correlation import HistogramData, SampledValue

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


def shift_fit_realisation(
    bins: NDArray,
    counts: NDArray,
    *,
    data: NDArray,
    sigma: NDArray
) -> FitResult[float]:
    sigma_is_cov = sigma.ndim > 1
    if sigma_is_cov:
        mask = np.isfinite(data) & np.isfinite(np.diag(sigma))
        sigma_masked = apply_bool_mask_ndim(sigma, mask)
    else:
        mask = np.isfinite(data) & np.isfinite(sigma)
        sigma_masked = sigma[mask]
    data_masked = data[mask]

    def model_masked(bins: NDArray, A: float, dz: float) -> NDArray:
        shifted = shift_histogram(bins, counts, A=A, dx=dz)
        return shifted[mask]

    # run fitting
    mids = (bins[1:] + bins[:-1]) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guess_A = 1.0 / np.trapz(counts, x=mids)
        guess_dz = 0.0
        popt, pcov, infodict, mesg, ier = curve_fit(
            model_masked, xdata=bins, ydata=data_masked,
            p0=[guess_A, guess_dz], sigma=sigma_masked,
            full_output=True)
    params = dict(zip(["A", "dz"], popt))
    ndof = np.count_nonzero(mask) - len(params)

    def chi_squared(params: Sequence[float]) -> float:
        prediction_masked = model_masked(bins, *params)
        r = prediction_masked - data_masked
        if sigma_is_cov:
            return r.T @ np.linalg.inv(sigma_masked) @ r
        else:
            return np.sum((r / sigma_masked) ** 2)

    return FitResult(ndof=ndof, chisq=chi_squared(popt), **params)


def shift_fit(
    data: RedshiftData,
    model: HistogramData,
    covariance: bool = False
) -> FitResult[SampledValue]:
    if not data.is_compatible(model):
        raise ValueError(
            "'data' and 'model' are not compatible in binning, spatial "
            "patches, or resampling method")
    if covariance:
        sigma = data.get_covariance().to_numpy()
    else:
        sigma = data.get_error().to_numpy()

    # fit the main values
    result = shift_fit_realisation(
        data.edges, model.data, data=data.data, sigma=sigma)

    # fit the data samples
    A = np.full(data.n_samples, fill_value=result["A"])
    dz = np.full(data.n_samples, fill_value=result["dz"])
    for i, (_data, _model) in enumerate(zip(data.samples, model.samples)):
        result_sample = shift_fit_realisation(
            data.edges, _model, data=_data, sigma=sigma)
        A[i] = result_sample["A"]
        dz[i] = result_sample["dz"]

    return FitResult(
        ndof=result.ndof,
        chisq=result.chisq,
        A=SampledValue(result["A"], A, method=data.method),
        dz=SampledValue(result["dz"], dz, method=data.method))
