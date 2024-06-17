"""This module implements a simple class to generate uniform randoms on a
rectangular footprint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from yaw.core.logging import TimedLog
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
        ra: float | NDArray[np.float64], dec: float | NDArray[np.float64]
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
        x: float | NDArray[np.float64], y: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
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
        if n_threads != 1:
            DeprecationWarning("'n_threads' is deprecated")
        seed = self.rng.spawn(1)[0]  # backwards compatibility
        msg = f"generate ({long_num_format(size)} uniform randoms)"
        with TimedLog(logger.info, msg):
            # build the output dataframe
            columns = ["ra", "dec"] if names is None else names
            if draw_from is not None:
                columns.extend(draw_from.keys())
            rand = pd.DataFrame(
                columns=["ra", "dec"] if names is None else names,
                index=pd.RangeIndex(0, size),
            )
            # generate the positions
            rng = np.random.default_rng(seed)
            x = rng.uniform(self.x_min, self.x_max, size)
            y = rng.uniform(self.y_min, self.y_max, size)
            ra_dec = UniformRandoms.cylinder2sky(x, y)
            rand["ra"] = ra_dec[:, 0]
            rand["dec"] = ra_dec[:, 1]
            # draw extra data
            if draw_from is not None:
                N = len(next(iter(draw_from.values())))
                draw_idx = rng.integers(0, N, size=size)
                for key, values in draw_from.items():
                    if not isinstance(values, np.ndarray):
                        raise TypeError(f"expected a numpy array for property '{key}")
                    if len(values) != N:
                        raise ValueError(
                            f"expected a {N} values to draw from for property '{key}'"
                        )
                    rand[key] = values[draw_idx]
        return rand
