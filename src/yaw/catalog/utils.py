from __future__ import annotations

from typing import Any, Generator, Literal

import numpy as np
from numpy.typing import NDArray


Tclosed = Literal["left", "right"]


def groupby_value(
    values: NDArray,
    **optional_arrays: NDArray | None,
) -> Generator[tuple[Any, dict[str, NDArray]], None, None]:
    idx_sort = np.argsort(values)
    values_sorted = values[idx_sort]
    uniques, _idx_split = np.unique(values_sorted, return_index=True)
    idx_split = _idx_split[1:]

    splitted_arrays = {}
    for name, array in optional_arrays.items():
        if array is not None:
            array_sorted = array[idx_sort]
            splitted_arrays[name] = np.split(array_sorted, idx_split)

    for i, value in enumerate(uniques):
        yield value, {name: splits[i] for name, splits in splitted_arrays.items()}


def groupby_binning(
    values: NDArray,
    binning: NDArray,
    closed: Tclosed = "left",
    **optional_arrays: NDArray | None,
) -> Generator[tuple[NDArray, dict[str, NDArray]], None, None]:
    binning = np.asarray(binning)
    bin_idx = np.digitize(values, binning, right=(closed == "right"))
    for i, bin_array in groupby_value(bin_idx, **optional_arrays):
        if i == 0 or i == len(binning):  # skip values outside of binning range
            continue
        yield binning[i - 1 : i + 1], bin_array


def logarithmic_mid(edges: NDArray) -> NDArray:
    log_edges = np.log10(edges)
    log_mids = (log_edges[:-1] + log_edges[1:]) / 2.0
    return 10.0**log_mids
