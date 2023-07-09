"""This module implements a simple class to generate uniform randoms on a
rectangular footprint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from yaw.core.logging import TimedLog
from yaw.core.parallel import POOL_SHARE, ParallelHelper, SharedArray
from yaw.core.utils import long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame

    from yaw.catalogs import BaseCatalog

__all__ = ["UniformRandoms"]


logger = logging.getLogger(__name__)


class UniformRandoms:
    """Generator for uniform randoms on a rectangular footprint.

    Generates points uniform in right ascension and declination. Additional
    features can be cloned by sampling values from an external data catalogue
    (e.g. spectroscopic or photometric redshifts).

    Internally uses cylindrical coordinates :math:`(x, y)`, which are an equal
    area projection. Point are generated in cylindrical coordinates and
    transformed back to spherical coordinates :math:`(\\alpha, \\delta)`:

    :math:`\\alpha \\leftrightarrow x \\quad \\sin{\\delta} \\leftrightarrow y`
    """

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        seed: int = 12345,
    ) -> None:
        """Create a new generator for a the given footprint.

        Args:
            ra_min (:obj:`float`):
                Minimum right ascenion to generate, in degrees.
            ra_max (:obj:`float`):
                Maximum right ascenion to generate, in degrees.
            dec_min (:obj:`float`):
                Minimum declination to generate, in degrees.
            dec_max (:obj:`float`):
                Maximum declination to generate, in degrees.
            seed (:obj:`int`, optional):
                Seed to use for the random generator.
        """
        self.x_min, self.y_min = self.sky2cylinder(ra_min, dec_min)
        self.x_max, self.y_max = self.sky2cylinder(ra_max, dec_max)
        self.rng = np.random.SeedSequence(seed)

    @classmethod
    def from_catalog(cls, cat: BaseCatalog, seed: int = 12345) -> UniformRandoms:
        """Create a new generator with a rectangular footprint obtained from the
        coordinate range of a given data catalogue.

        Args:
            cat (:obj:`yaw.catalogs.BaseCatalog`):
                Catalog instance from which the right ascension and declination
                range is computed.
            seed (:obj:`int`, optional):
                Seed to use for the random generator.
        """
        return cls(
            np.rad2deg(cat.ra.min()),
            np.rad2deg(cat.ra.max()),
            np.rad2deg(cat.dec.min()),
            np.rad2deg(cat.dec.max()),
            seed=seed,
        )

    @staticmethod
    def sky2cylinder(
        ra: float | NDArray[np.float_], dec: float | NDArray[np.float_]
    ) -> NDArray:
        """Conversion from spherical to cylindrical coordinates.

        Args:
            ra (:obj:`float`, :obj:`NDArray`):
                Right ascension(s) to convert to cylindrical coordinates.
            dec (:obj:`float`, :obj:`NDArray`):
                Right ascension(s) to convert to cylindrical coordinates.

        Returns:
            :obj:`NDArray`:
                Array with of points in cylindrical coordinates of shape
                `(N, 2)`.
        """
        x = np.deg2rad(ra)
        y = np.sin(np.deg2rad(dec))
        return np.transpose([x, y])

    @staticmethod
    def cylinder2sky(
        x: float | NDArray[np.float_], y: float | NDArray[np.float_]
    ) -> float | NDArray[np.float_]:
        """Conversion from cylindrical to spherical coordinates.

        Args:
            x (:obj:`float`, :obj:`NDArray`):
                `x`-coordinate(s) to convert to spherical coordinates.
            y (:obj:`float`, :obj:`NDArray`):
                `y`-coordinate(s) to convert to spherical coordinates.

        Returns:
            :obj:`NDArray`:
                Array with of points in spherical coordinates of shape `(N, 2)`.
        """
        ra = np.rad2deg(x)
        dec = np.rad2deg(np.arcsin(y))
        return np.transpose([ra, dec])

    def generate(
        self,
        size: int,
        names: list[str, str] | None = None,
        draw_from: dict[str, NDArray] | None = None,
        n_threads: int = 1,
    ) -> DataFrame:
        """Generate new random points.

        Generate a specified number of points, additionally draw extra data
        features form a list of input values. Results are returned in a data
        frame.

        Args:
            size (:obj:`int`):
                Number of random points to generate.
            name (:obj:`tuple[str, str]`, optional):
                Name of the right ascension and declination columns in the
                output data frame. Default is ``ra`` and ``dec``.
            draw_from (:obj:`dict[str, NDArray]`, optional):
                Dictionary of data arrays. If provided, a random sample (with
                repetition) is drawn from these arrays and assigned to the
                output data frame. The dictionary keys are used to name the
                columns in the output.
            n_threads (:obj:`int`, optional):
                Generate data in parallel using subprocesses, default is
                parallel processing disabled.

                .. deprecated:: 2.3.2
                    No performance gain observed. May be removed in a future
                    version.

        Returns:
            :obj:`pandas.DataFrame`:
                Data frame with uniform random coordinates and optionally
                additional features draw from input data.
        """
        # seeds for threads
        if size <= 100 * n_threads:  # there is some kind of floor
            n_threads = 1
        seeds = self.rng.spawn(n_threads)
        # distribute load
        idx_break = np.linspace(0, size, n_threads + 1).astype(np.int_)
        start = idx_break[:-1]
        end = idx_break[1:]

        # create output arrays
        shares = dict(
            ra=SharedArray.empty((size,), "f8"), dec=SharedArray.empty((size,), "f8")
        )
        if draw_from is not None:
            shares["in"] = {
                key: SharedArray.from_numpy(array) for key, array in draw_from.items()
            }
            shares["out"] = {
                key: SharedArray.empty((size,), array.dtype)
                for key, array in draw_from.items()
            }
        if names is None:
            names = ["ra", "dec"]

        # process
        msg = f"generate ({long_num_format(size)} uniform randoms)"
        with TimedLog(logger.info, msg):
            with ParallelHelper(_generate_uniform_randoms, n_threads) as pool:
                pool.shares = shares  # assign in bulk
                pool.add_constant(self)
                pool.add_iterable(seeds)
                pool.add_iterable(start)
                pool.add_iterable(end)
                pool.result()  # output direclty pasted into shared arrays
                # convert to dataframe
                rand = pd.DataFrame(
                    {
                        names[0]: pool.shares["ra"].to_numpy(copy=True),
                        names[1]: pool.shares["dec"].to_numpy(copy=True),
                    }
                )
                if draw_from is not None:
                    for col, data in pool.shares["out"].items():
                        rand[col] = data.to_numpy(copy=True)
                # will delete shared arrays after this
        return rand


def _generate_uniform_randoms(
    inst: UniformRandoms, seed: np.random.SeedSequence, start: int, end: int
) -> int:
    rng = np.random.default_rng(seed)
    size = end - start
    ra = POOL_SHARE["ra"].to_numpy()
    dec = POOL_SHARE["dec"].to_numpy()
    # generate positions
    x = rng.uniform(inst.x_min, inst.x_max, size)
    y = rng.uniform(inst.y_min, inst.y_max, size)
    ra_dec = UniformRandoms.cylinder2sky(x, y)
    ra[start:end] = ra_dec[:, 0]
    dec[start:end] = ra_dec[:, 1]
    # generate random draw
    if "in" in POOL_SHARE:
        N = None
        for col, data in POOL_SHARE["in"].items():
            if N is None:
                if len(data.shape) > 1:
                    raise ValueError("data to draw from must be 1-dimensional")
                N = len(data)
            else:
                if len(data) != N:
                    raise ValueError("length of columns to draw from does not match")
        draw_idx = rng.integers(0, N, size=size)
        # draw and insert data
        for col, data in POOL_SHARE["in"].items():
            rand = POOL_SHARE["out"][col].to_numpy()
            rand[start:end] = data.to_numpy()[draw_idx]
    return size
