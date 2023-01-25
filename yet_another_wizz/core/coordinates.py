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
    ra_dec_rad = np.atleast_2d(RA_DEC)
    ra = ra_dec_rad[:, 0]
    dec = ra_dec_rad[:, 1]
    cos_dec = np.cos(dec)
    # transform
    pos_sphere = np.empty((len(ra_dec_rad), 3))
    pos_sphere[:, 0] = np.cos(ra) * cos_dec
    pos_sphere[:, 1] = np.sin(ra) * cos_dec
    pos_sphere[:, 2] = np.sin(dec)
    return np.squeeze(pos_sphere)


def position_sphere2sky(xyz: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Maps Euclidean coordinates (x, y, z) onto celestial coordinates
    (RA, Dec) in radians.
    """
    # unpack data and compute intermediate values
    xyz = np.atleast_2d(xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    r_d3 = np.sqrt(x2 + y2 + z2)
    r_d2 = np.sqrt(x2 + y2)
    # transform
    pos_sky = np.empty((len(xyz), 2))
    pos_sky[:, 0] = np.sign(y) * np.arccos(x / r_d2)
    pos_sky[:, 1] = np.sign(y) * np.arcsin(z / r_d3)
    return np.squeeze(pos_sky)


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
