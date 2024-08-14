from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


PRECISION = 10


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"[:width]
    if "nan" in string or "inf" in string:
        string = f"{string.strip():>{width}s}"
    return string


def write_header(f, description, header):
    f.write(f"{description}\n")
    line = " ".join(f"{h:>{PRECISION}s}" for h in header)
    f.write(f"# {line[2:]}\n")


def write_data(path: Path, description: str, zleft: NDArray, zright: NDArray, data: NDArray, error: NDArray):
    with path.open("w") as f:
        header = ["z_low", "z_high", "nz", "nz_err"]
        write_header(f, description, header)

        for values in zip(zleft, zright, data, error):
            formatted = [format_float_fixed_width(value, PRECISION) for value in values]
            f.write(" ".join(formatted) + "\n")


def write_samples(path: Path, description: str, zleft: NDArray, zright: NDArray, samples: NDArray, method: str):
    with path.open("w") as f:
        header = ["z_low", "z_high"]
        header.extend(f"{method[:4]}_{i}" for i in range(len(samples)))
        write_header(f, description, header)

        for zleft, zright, samples in zip(zleft, zright, samples.T):
            formatted = [
                format_float_fixed_width(zleft, PRECISION),
                format_float_fixed_width(zright, PRECISION),
            ]
            formatted.extend(
                format_float_fixed_width(value, PRECISION) for value in samples
            )
            f.write(" ".join(formatted) + "\n")


def write_covariance(path: Path, description: str, covariance: NDArray):
    with path.open("w") as f:
        f.write(f"{description}\n")

        for row in covariance:
            for value in row:
                f.write(f"{value: .{PRECISION - 3}e} ")
            f.write("\n")


def load_data(path: Path) -> tuple[NDArray, NDArray]:
    zleft, zright, data, _ = np.loadtxt(path).T
    edges = np.append(zleft, zright[-1])
    return edges, data


def load_samples(path: Path) -> tuple[NDArray, str]:
    with path.open() as f:
        for line in f.readlines():
            if "z_low" in line:
                line = line[2:].strip("\n")  # remove leading '# '
                header = [col for col in line.split(" ") if len(col) > 0]
                break
        else:
            raise ValueError("sample file header misformatted")

    method_key, _ = header[-1].rsplit("_", 1)
    for method in Tmethod.__args__:
        if method.startswith(method_key):
            break
    else:
        raise ValueError(f"invalid sampling method key '{method_key}'")

    samples = np.loadtxt(path).T[2:]  # remove binning columns
    return samples, method
