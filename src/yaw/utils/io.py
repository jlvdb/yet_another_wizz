from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from h5py import Group
    from numpy.typing import NDArray

HDF_COMPRESSION = dict(fletcher32=True, compression="gzip", shuffle=True)

PRECISION = 10


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"
    if "nan" in string or "inf" in string:
        string = f"{string.rstrip():>{width}s}"

    num_digits = len(string.split(".")[0])
    return string[: max(width, num_digits)]


def create_columns(columns: list[str], closed: str) -> list[str]:
    if closed == "left":
        all_columns = ["[z_low", "z_high)"]
    else:
        all_columns = ["(z_low", "z_high]"]
    all_columns.extend(columns)
    return all_columns


def write_header(f, description, columns) -> None:
    line = " ".join(f"{col:>{PRECISION}s}" for col in columns)

    f.write(f"# {description}\n")
    f.write(f"#{line[1:]}\n")


def load_header(path: Path) -> tuple[str, list[str], str]:
    def unwrap_line(line):
        return line.lstrip("#").strip()

    with path.open() as f:
        description = unwrap_line(f.readline())
        columns = unwrap_line(f.readline()).split()

    closed = "left" if columns[0][0] == "[" else "right"
    return description, columns, closed


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
        write_header(f, description, create_columns(["nz", "nz_err"], closed))

        for values in zip(zleft, zright, data, error):
            formatted = [format_float_fixed_width(value, PRECISION) for value in values]
            f.write(" ".join(formatted) + "\n")


def load_data(path: Path) -> tuple[NDArray, str, NDArray]:
    _, _, closed = load_header(path)

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
    closed: str,
) -> None:
    with path.open("w") as f:
        sample_columns = [f"jack_{i}" for i in range(len(samples))]
        write_header(f, description, create_columns(sample_columns, closed))

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
        f.write(f"# {description}\n")

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
        tag = source["version"][()]
        return tag.decode("utf-8")

    except KeyError:
        return "2.x.x"


def is_legacy_dataset(source: Group) -> bool:
    return "version" not in source
