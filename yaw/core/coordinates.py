from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def position_sky2sphere(RA_DEC: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Maps celestial coordinates (radians) onto a unit-sphere in three
    dimensions (x, y, z).
    """
    # unpack data and compute intermediate values
    ra, dec = np.atleast_2d(RA_DEC).T
    cos_dec = np.cos(dec)
    # transform
    x = np.cos(ra) * cos_dec
    y = np.sin(ra) * cos_dec
    z = np.sin(dec)
    xyz = np.transpose([x, y, z])
    return np.squeeze(xyz)


def position_sphere2sky(xyz: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Maps Euclidean coordinates (x, y, z) onto celestial coordinates
    (RA, Dec) in radians.
    """
    def sgn(x):
        return np.where(x == 0, 1.0, np.sign(x))

    # unpack data and compute intermediate values
    x, y, z = np.atleast_2d(xyz).T
    r_d2 = np.sqrt(x*x + y*y)
    # transform
    ra = np.arccos(x / r_d2) * sgn(y) % (2.0*np.pi)
    ra[np.isnan(ra)] = 0.0
    dec = np.arcsin(z)
    rd = np.transpose([ra, dec])
    return np.squeeze(rd)


def distance_sky2sphere(dist_sky: ArrayLike) -> ArrayLike:
    """
    Converts angular separation in celestial coordinates (radians) to the
    Euclidean distance in (x, y, z) space.
    """
    return 2.0 * np.sin(dist_sky / 2.0)


def distance_sphere2sky(dist_sphere: ArrayLike) -> ArrayLike:
    """
    Converts Euclidean distance in (x, y, z) space to angular separation in
    celestial coordinates (radians).
    """
    return 2.0 * np.arcsin(dist_sphere / 2.0)
