from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame


class UniformRandoms:

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        seed: int = 12345
    ) -> None:
        self.x_min, self.y_min = self.sky2cylinder(ra_min, dec_min)
        self.x_max, self.y_max = self.sky2cylinder(ra_max, dec_max)
        self.rng = np.random.SeedSequence(seed)

    @classmethod
    def from_catalogue(cls, cat) -> UniformRandoms:
        raise NotImplementedError

    @staticmethod
    def sky2cylinder(
        ra: float | NDArray[np.float_],
        dec: float | NDArray[np.float_]
    ) -> NDArray:
        x = np.deg2rad(ra)
        y = np.sin(np.deg2rad(dec))
        return np.transpose([x, y])
 
    @staticmethod
    def cylinder2sky(
        x: float | NDArray[np.float_],
        y: float | NDArray[np.float_]
    ) -> float | NDArray[np.float_]:
        ra = np.rad2deg(x)
        dec = np.rad2deg(np.arcsin(y))
        return np.transpose([ra, dec])

    def generate(
        self,
        size: int,
        names: list[str, str] | None = None,
        draw_from: dict[str, NDArray] | None = None,
        n_threads: int = 1
    ) -> DataFrame:
        seeds = self.rng.spawn(n_threads)
        if size <= 100 * n_threads:
            n_threads = 1
        sizes = np.diff(np.linspace(0, size, n_threads+1).astype(np.int_))
        args = []
        for i in range(n_threads):
            args.append([self, seeds[i], sizes[i], names, draw_from])
        with multiprocessing.Pool(n_threads) as pool:
            chunks = pool.starmap(_generate_uniform_randoms, args)
        return pd.concat(chunks)


def _generate_uniform_randoms(
    inst: UniformRandoms,
    seed: np.random.SeedSequence,
    size: int,
    names: list[str, str] | None = None,
    draw_from: dict[str, NDArray] | None = None
) -> DataFrame:
    rng = np.random.default_rng(seed)
    if names is None:
        names = ["ra", "dec"]
    # generate positions
    x = np.random.uniform(inst.x_min, inst.x_max, size)
    y = np.random.uniform(inst.y_min, inst.y_max, size)
    ra, dec = UniformRandoms.cylinder2sky(x, y).T
    rand = DataFrame({names[0]: ra, names[1]: dec})
    # generate random draw
    if draw_from is not None:
        N = None
        for col in draw_from.values():
            if N is None:
                if len(col.shape) > 1:
                    raise ValueError("data to draw from must be 1-dimensional")
                N = len(col)
            else:
                if len(col) != N:
                    raise ValueError(
                        "length of columns to draw from does not match")
        draw_idx = rng.integers(0, N, size=size)
        # draw and insert data
        for key, col in draw_from.items():
            rand[key] = col[draw_idx]
    return rand
