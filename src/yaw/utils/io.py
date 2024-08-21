from __future__ import annotations

from pathlib import Path

import numpy as np
from h5py import Group
from numpy.typing import NDArray

HDF_COMPRESSION = dict(fletcher32=True, compression="gzip", shuffle=True)

PRECISION = 10


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"[:width]
    if "nan" in string or "inf" in string:
        string = f"{string.strip():>{width}s}"
    return string


def write_header(f, description, columns, extra_info: str | None = None) -> None:
    line = " ".join(f"{col:>{PRECISION}s}" for col in columns)

    f.write(f"# {description}\n")
    f.write(f"#{line[1:]}\n")
    if extra_info is not None:
        f.write(f"# {extra_info}\n")


def load_header(path: Path) -> dict[str, str | list[str]]:
    def unwrap_line(line):
        return line.lstrip("#").strip()

    header = dict()
    with path.open() as f:
        header["description"] = unwrap_line(f.readline())
        header["columns"] = unwrap_line(f.readable()).split()
        if (extra_info := f.readable()).startswith("#"):
            header["extra_info"] = unwrap_line(extra_info)
    return header


def write_data(
    path: Path,
    description: str,
    *,
    zleft: NDArray,
    zright: NDArray,
    data: NDArray,
    error: NDArray,
    closed: str,
) -> None:
    with path.open("w") as f:
        columns = ["z_low", "z_high", "nz", "nz_err"]
        write_header(f, description, columns, extra_info=f"interval closed: {closed}")

        for values in zip(zleft, zright, data, error):
            formatted = [format_float_fixed_width(value, PRECISION) for value in values]
            f.write(" ".join(formatted) + "\n")


def load_data(path: Path) -> tuple[NDArray, str, NDArray]:
    header = load_header(path)
    _, closed = header["extra_info"].split(": ")

    zleft, zright, data, _ = np.loadtxt(path).T
    edges = np.append(zleft, zright[-1])
    return edges, closed, data


def write_samples(
    path: Path,
    description: str,
    *,
    zleft: NDArray,
    zright: NDArray,
    samples: NDArray,
) -> None:
    with path.open("w") as f:
        columns = ["z_low", "z_high"]
        columns.extend(f"jack_{i}" for i in range(len(samples)))
        write_header(f, description, columns)

        for zleft, zright, samples in zip(zleft, zright, samples.T):
            formatted = [
                format_float_fixed_width(zleft, PRECISION),
                format_float_fixed_width(zright, PRECISION),
            ]
            formatted.extend(
                format_float_fixed_width(value, PRECISION) for value in samples
            )
            f.write(" ".join(formatted) + "\n")


def load_samples(path: Path) -> NDArray:
    return np.loadtxt(path).T[2:]  # remove binning columns


def write_covariance(path: Path, description: str, *, covariance: NDArray) -> None:
    with path.open("w") as f:
        f.write(f"{description}\n")

        for row in covariance:
            for value in row:
                f.write(f"{value: .{PRECISION - 3}e} ")
            f.write("\n")


# NOTE: load_covariance() not required


def write_version_tag(dest: Group) -> None:
    from yaw._version import __version__

    dest.create_dataset("version", data=__version__)


def load_version_tag(source: Group) -> str:
    try:
        return source["version"][()]
    except KeyError:
        return "2.x.x"


def is_legacy_dataset(source: Group) -> bool:
    return "version" not in source
