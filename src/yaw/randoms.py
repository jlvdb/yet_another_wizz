"""
Implements random generators that can be sampled to create random catalogs for
correlation measurements.

The generators create uniform random coordinates (with optional constraints) and
can additionally draw weights or redshifts from a set of observed values.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from yaw.datachunk import DataChunk, DataChunkInfo, HandlesDataChunk, TypeDataChunk

HEALPY_ENABLED = False
"""Healpix-based randoms are enabled if healpy can be imported."""
try:
    import healpy

    HEALPY_ENABLED = True

except ImportError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BoxRandoms",
    "HealPixRandoms",
]


class RandomsBase(HandlesDataChunk):
    """Meta-class for all random point generators."""

    @abstractmethod
    def __init__(
        self,
        *args,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        seed: int = 12345,
        **kwargs,
    ) -> None:
        self._chunk_info = DataChunkInfo(
            has_weights=weights is not None,
            has_redshifts=redshifts is not None,
        )
        self.reseed(seed)
        self.weights = weights
        self.redshifts = redshifts
        self.data_size = self.get_data_size()

    def get_data_size(self) -> int:
        """
        Get the number attached data samples to draw from.

        Checks the length of the :attr:`weights` and :attr:`redshifts` and
        returns their length. If neither are defined, returns -1.

        Returns:
            Number of observations or -1.

        Raises:
            ValueError:
                If the lengths of the arrays do not match.
        """
        if self.weights is None and self.redshifts is None:
            return -1
        elif self.weights is None:
            return len(self.redshifts)
        elif self.redshifts is None:
            return len(self.weights)

        if len(self.weights) != len(self.redshifts):
            raise ValueError(
                "number of 'weights' and 'redshifts' to draw from does not match"
            )
        return len(self.weights)

    def reseed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def _draw_coords(self, probe_size: int) -> tuple[NDArray, NDArray]:
        """
        Draw a number of random uniform coordinates in radian.

        Args:
            probe_size:
                Number of points to draw with repetition from the data samples.

        Returns:
            Tuple of arrays containing random right ascensions and declinations.
        """
        pass

    def _draw_attributes(self, probe_size: int) -> dict[str, NDArray]:
        """
        Draw a number of samples from attached data samples.

        Args:
            probe_size:
                Number of points to draw with repetition from the data samples.

        Returns:
            Dictionary with optional keys ``weights`` and ``redshifts`` with
            drawn samples.
        """
        if self.data_size == -1:
            return dict()

        data = dict()
        idx = self.rng.integers(0, self.data_size, size=probe_size)
        if self.has_weights:
            data["weights"] = self.weights[idx]
        if self.has_redshifts:
            data["redshifts"] = self.redshifts[idx]
        return data

    def __call__(self, probe_size: int) -> TypeDataChunk:
        """
        Draw a new sample of uniform random points.

        Args:
            probe_size:
                Number of points to generate.

        Returns:
            Dictionary of arrays, always contains keys ``ra`` and ``dec`` for
            coordinates of random points in radian. Optionally contains
            ``weights`` and/or ``redshifts`` if data as been provided to sample
            from.
        """
        ra, dec = self._draw_coords(probe_size)
        optionals = self._draw_attributes(probe_size)
        _, chunk = DataChunk.create(
            ra, dec, **optionals, degrees=False, chkfinite=False
        )
        return chunk


class BoxRandoms(RandomsBase):
    """
    Generates random points within a right ascension / declination window.

    Generators are used with the :meth:`~yaw.Catalog.from_random` method to
    create a catalog with uniformly distributed random coordiantes. Additional
    redshifts or weights (e.g. from an observed data sample) may be attached to
    randomly sample from their distribution.

    Call instance to generate random points.

    Args:
        ra_min:
            The lower limit of the right ascension in degrees.
        ra_max:
            The upper limit of the right ascension in degrees.
        dec_min:
            The lower limit of the declination in degrees.
        dec_max:
            The upper limit of the declination in degrees.
        weights:
            Optional array of weights to draw from.
        redshifts:
            Optional array of redshifts to draw from.
        seed:
            Integer number from which the random seed is generated.

    Attributes:
        weights:
            Optional array of weights to draw from.
        redshifts:
            Optional array of redshifts to draw from.
    """

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        seed: int = 12345,
    ) -> None:
        super().__init__(weights=weights, redshifts=redshifts, seed=seed)

        self.x_min, self.y_min = self._sky2cylinder(
            np.deg2rad(ra_min), np.deg2rad(dec_min)
        )
        self.x_max, self.y_max = self._sky2cylinder(
            np.deg2rad(ra_max), np.deg2rad(dec_max)
        )

    def __repr__(self) -> str:
        string = repr(self._chunk_info)
        string = string.lstrip(str(type(self._chunk_info)))
        return f"{type(self).__name__}{string}"

    def _sky2cylinder(self, ra: NDArray, dec: NDArray) -> tuple[NDArray, NDArray]:
        x = ra
        y = np.sin(dec)
        return x, y

    def _cylinder2sky(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        ra = x
        dec = np.arcsin(y)
        return ra, dec

    def _draw_coords(self, probe_size: int) -> tuple[NDArray, NDArray]:
        x = self.rng.uniform(self.x_min, self.x_max, probe_size)
        y = self.rng.uniform(self.y_min, self.y_max, probe_size)
        return self._cylinder2sky(x, y)


class HealPixRandoms(RandomsBase):
    """
    Generates random points within a `HealPix` mask.

    The input mask can either be interpreted as a boolean mask or a probability
    map that indicates the probability at which random points should be
    generated in a given pixel. Requires installing ``healpy``.

    Call instance to generate random points.

    .. Caution::
        To improve performance, this method does not create continuous random
        points, but randomly draws pixel center coordinates from the highest
        possible mask resolution supported by `HealPix`. This corresponds to
        a resolution of about :math:`2500` pixels/arcsec.

    Args:
        pix_values:
            Array with `HealPix` mask values.
        nested:
            Whether the input mask is in the `nested` format.
        is_mask:
            Whether the input mask should be interpreted as binary mask or as
            probability map (default).
        weights:
            Optional array of weights to draw from.
        redshifts:
            Optional array of redshifts to draw from.
        seed:
            Integer number from which the random seed is generated.

    Attributes:
        weights:
            Optional array of weights to draw from.
        redshifts:
            Optional array of redshifts to draw from.
        nside:
            The `HealPix` ``nside`` value of the input mask.

    Raises:
        ImportError:
            If ``healpy`` is not installed.
    """

    def __init__(
        self,
        pix_values: NDArray,
        *,
        nested: bool = False,
        is_mask: bool = False,
        weights=None,
        redshifts=None,
        seed=12345,
    ):
        if not HEALPY_ENABLED:
            raise ImportError("could not import optional dependency 'healpy'")

        super().__init__(weights=weights, redshifts=redshifts, seed=seed)

        values = np.asarray(pix_values, dtype=np.float64)
        self.nside = healpy.npix2nside(len(values))
        if np.any(values < 0.0):
            raise ValueError("pixel values must be positive for random generation")

        # check which of the pixels are masked
        if not nested:
            values = healpy.reorder(values, inp="RING", out="NESTED")
        self._ipix_unmasked = np.nonzero(values)[0]

        # compute the probability of drawing from unmasked pixels
        if is_mask:
            self._probability = None
        else:
            masked_values = values[self._ipix_unmasked]
            self._probability = masked_values / masked_values.sum()

    def _draw_coords(self, probe_size: int) -> tuple[NDArray, NDArray]:
        """
        The general idea is to generate a list of pixels to draw coordinates
        from. Then go to that pixel, view it at the highest possible resolution
        and draw a random sub-pixel and use its center coordinate.
        """
        MAX_ORDER = 29
        MAX_NSIDE = 2**MAX_ORDER  # sample random pixel IDs at this resolution

        # generate list of pixel IDs to draw from (factoring in pixel weight)
        ipix_draw = np.random.choice(
            self._ipix_unmasked,
            size=probe_size,
            p=self._probability,
        )
        # transform pixel ID to resolution of MAX_ORDER (using nested convention)
        order = healpy.nside2order(self.nside)
        scale = 4 ** (MAX_ORDER - order)
        ipix_scaled = ipix_draw * scale

        # draw random pixel IDs at maximum resolution and get center coordinates
        ipix_rand = ipix_scaled + self.rng.integers(0, scale, size=probe_size)
        ra, dec = healpy.pix2ang(
            nside=MAX_NSIDE, ipix=ipix_rand, nest=True, lonlat=True
        )
        return np.deg2rad(ra), np.deg2rad(dec)
