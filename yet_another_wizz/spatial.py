import numpy as np
import pandas as pd
from astropy import units
from astropy.cosmology import FLRW, default_cosmology, Planck15
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import cKDTree, minkowski_distance


class FastSeparation2Angle(object):

    _is_comoving = True

    """
    fastSeparation2Angle(zmin=0.001, zmax=100.0, nlogsamples=100, units=False)

    fast conversion from transverse comoving separation to angles

    This class offers two methods to compute the separation angle between two
    points on the sky corresponding to a given transverse comoving distance
    in kpc at a given redshift. Next to the exact calculation using the
    astropy.cosmology module it offers approximations using a cubic spline fit
    to precomputed values of from a given cosmological model.

    Parameters
    ----------
    zmin : positive float
        Minimum redshift at which data for the spline fit is computed.
    zmax : positive float
        Maximum redshift at which data for the spline fit is computed.
    nlogsamples : positive integer
        Number of logarithmically spaced sampling points to wich the spline
        is fitted.

    """

    # use the default cosmolgy of astropy
    cosmology = Planck15

    def __init__(self, zmin=0.001, zmax=100.0, nlogsamples=100, units=False):
        if zmax <= zmin:
            raise ValueError("zmax must be larger than zmin")
        # compute a logarighmically spaced redshift sampling
        self.z_log_min = np.log10(zmin)
        self.z_log_max = np.log10(zmax)
        self.nlogsamples = nlogsamples
        self.z_log_samples = np.logspace(
            self.z_log_min, self.z_log_max, self.nlogsamples)
        # fit the cubic spline
        self._fit_spline()

    def set_comoving(self):
        self._is_comoving = True
        self._fit_spline()

    def set_physical(self):
        self._is_comoving = False
        self._fit_spline()

    def set_cosmology(self, cosmology):
        """
        Compute the L**p distance between two arrays.

        Parameters
        ----------
        x : astropy.cosmology.core.FLRW subclass or string
            class that provides methods to compute the number of arcseconds
            per kpc (comoving), or a string specifing a predefined cosmology
            in astropy.

        Examples
        --------
        >>> get_angle = fastDang2Angle()
        >>> get_angle.set_cosmology("WMAP7")
        >>> get_angle.cosmology
        FlatLambdaCDM(name="WMAP7", H0=70.4 km / (Mpc s), Om0=0.272,
        Tcmb0=2.725 K, Neff=3.04, m_nu=[0. 0. 0.] eV, Ob0=0.0455)

        """
        if type(cosmology) is str:
            if cosmology == "default":
                cosmology = "Planck15"
            self.cosmology = \
                default_cosmology.get_cosmology_from_string(cosmology)
        elif not issubclass(type(cosmology), FLRW):
            raise TypeError(
                "cosmology must be subclass of type %s" % str(FLRW))
        else:
            self.cosmology = cosmology
        self._fit_spline()

    def _fit_spline(self):
        """
        Update the internal cubic spline fit to samples of the exact evaluation
        of the separation angle on the sky between two points corresponding to
        1 kpc transverse separation for at a given redshift.

        """
        # use the redshift sample compute at instantiation to fit cubic spline
        self.spline = InterpolatedUnivariateSpline(
            self.z_log_samples, self.exact(self.z_log_samples, 1.0), k=3)

    def exact(self, z, scale_kpc):
        """
        Compute the separation angle on the sky between two points
        corresponding to a transverse comoving separation at a given redshift.

        Parameters
        ----------
        z : array_like
            The redshift at which the separation angle is computed.
        scale_kpc : positive float
            The transverse comoving separation in kpc.

        Returns
        -------
        results : array_like or astropy.units.quantity.Quantity
            The separation angle in degrees. If self.units is true, the result
            is given with an astropy unit

        """
        # convert to an astropy unit object
        r_kpc = scale_kpc * units.kpc
        # compute the separation angle in arcseconds
        if self._is_comoving:
            arcsec = self.cosmology.arcsec_per_kpc_comoving(z) * r_kpc
        else:
            arcsec = self.cosmology.arcsec_per_kpc_proper(z) * r_kpc
        return arcsec.to(units.deg).value

    def fast(self, z, scale_kpc):
        """
        Fast computation of the separation angle on the sky between two points
        corresponding to a transverse comoving separation at a given redshift.
        Uses a spline fit to the exact result to improve performace at the cost
        of accuracy.

        Parameters
        ----------
        z : array_like
            The redshift at which the separation angle is computed.
        scale_kpc : float
            The projected separation in kpc.

        Returns
        -------
        results : float or astropy.units.quantity.Quantity
            The separation angle in degrees. If self.units is true, the result
            is given with an astropy unit

        Notes
        -----
        Accuracy loss is usually negligible but evaluation can be serveral
        ten times faster than compared to self.exact.

        """
        # evaluate the spline fit that gives the angular size of 1 kpc comoving
        # at the given redshift and scale it to the input scale.
        return self.spline(z) * scale_kpc


class SphericalKDTree(object):
    """
    SphericalKDTree(RA, DEC, leaf_size=16)

    A binary search tree based on scipy.spatial.cKDTree that works with
    celestial coordinates. Provides methods to find pairs within angular
    apertures (ball) and annuli (shell). Data is internally represented on a
    unit-sphere in three dimensions (x, y, z).

    Parameters
    ----------
    RA : array_like
        List of right ascensions in degrees.
    DEC : array_like
        List of declinations in degrees.
    leafsize : int
        The number of points at which the algorithm switches over to
        brute-force.
    """

    def __init__(self, RA, DEC, leafsize=16):
        # convert angular coordinates to 3D points on unit sphere
        pos_sphere = self._position_sky2sphere(RA, DEC)
        self.tree = cKDTree(pos_sphere, leafsize)

    @staticmethod
    def _position_sky2sphere(RA, DEC):
        """
        Maps celestial coordinates onto a unit-sphere in three dimensions
        (x, y, z).

        Parameters
        ----------
        RA : float or array_like
            Single or list of right ascensions in degrees.
        DEC : float or array_like
            Single or list of declinations in degrees.

        Returns
        -------
        pos_sphere : array like
            Data points (x, y, z) representing input points on the unit-sphere,
            shape of output is (3,) for a single input point or (N, 3) for a
            set of N input points.
        """
        ras_rad = np.deg2rad(RA)
        decs_rad = np.deg2rad(DEC)
        try:
            pos_sphere = np.empty((len(RA), 3))
        except TypeError:
            pos_sphere = np.empty((1, 3))
        cos_decs = np.cos(decs_rad)
        pos_sphere[:, 0] = np.cos(ras_rad) * cos_decs
        pos_sphere[:, 1] = np.sin(ras_rad) * cos_decs
        pos_sphere[:, 2] = np.sin(decs_rad)
        return np.squeeze(pos_sphere)

    @staticmethod
    def _distance_sky2sphere(dist_sky):
        """
        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.

        Parameters
        ----------
        dist_sky : float or array_like
            Single or list of separations in celestial coordinates.

        Returns
        -------
        dist_sphere : float or array_like
            Celestial separation converted to (x, y, z) Euclidean distance.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere):
        """
        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.

        Parameters
        ----------
        dist_sphere : float or array_like
            Single or list of Euclidean distances in (x, y, z) space.

        Returns
        -------
        dist_sky : float or array_like
            Euclidean distance converted to celestial angular separation.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def query_radius(self, RA, DEC, r):
        """
        Find all data points within an angular aperture r around a reference
        point with coordiantes (RA, DEC) obeying the spherical geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        r : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with sepration < r.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        point_sphere = self._position_sky2sphere(RA, DEC)
        # find all points that lie within r
        r_sphere = self._distance_sky2sphere(r)
        idx = self.tree.query_ball_point(point_sphere, r_sphere)
        # compute pair separation
        dist_sphere = minkowski_distance(self.tree.data[idx], point_sphere)
        dist = self._distance_sphere2sky(dist_sphere)
        return idx, dist

    def query_shell(self, RA, DEC, rmin, rmax):
        """
        Find all data points within an angular annulus rmin <= r < rmax around
        a reference point with coordiantes (RA, DEC) obeying the spherical
        geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        rmin : float
            Minimum separation of data points from the reference point.
        rmax : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with rmin <= sepration < rmax.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        # find all points that lie within rmax
        idx, dist = self.query_radius(RA, DEC, rmax)
        # remove pairs with r >= rmin
        dist_mask = dist >= rmin
        idx = np.compress(dist_mask, idx)
        dist = np.compress(dist_mask, dist)
        return idx, dist


def count_pairs(
        group_reference, group_other, rlimits, comoving=False,
        cosmology=None, inv_distance_weight=True):
    """
    Count pairs between a reference and an unknown data catalouge with a
    constant physical or comoving separation r_min <= r < r_max using k-nearest
    neighbour search. Individual object weights from both catalogues and an
    inverse distance weight can be included.

    Parameters
    ----------
    group_reference : tuple (as returned by pandas.DataFrame.groupby)
        Reference object catalogue around which the other catalogue is queried,
        must contain a pandas.DataFrame with keys 'RA' (right ascension), 'DEC'
        (declination), 'z' (redshift) and optionally 'weights' (object weights)
    group_other : tuple (as returned by pandas.DataFrame.groupby)
        Other catalogue from which pairs are selected using a k-nearest
        neighbour tree, must contain a pandas.DataFrame with keys 'RA' (right
        ascension), 'DEC' (declination) and optionally 'weights' (object
        weights)
    rlimits : tuple
        Tuple of minimum and maximum projected comoving/physical distance used
        to select object pairs.
    comoving : bool
        Whether the rlimits are comoving or physical projected distances.
    cosmology : astropy.cosmology
        An astropy cosmology instance used for distance calculations.
    inv_distance_weight : bool
        Whether or not to use the inverse distance of two partners as
        additional weight for the pair.

    Returns
    -------
    pair_counts : pandas.DataFrame
        DataFrame with reference catalogue indices and sum of pair weights
        associated with each reference object.
    """
    # unpack the pandas groups and dictionary
    region_idx, data_reference = group_reference
    region_idx, data_other = group_other
    try:
        weights_other = data_other.weights.to_numpy()
    except Exception:  # default to unity weight
        weights_other = np.ones(len(data_other))
    # initialize fast angular diameter distance calculator
    get_angle = FastSeparation2Angle()
    get_angle.set_cosmology(cosmology)
    if comoving:
        get_angle.set_comoving()
    else:
        get_angle.set_physical()
    # compute annuli
    ang_min = get_angle.fast(data_reference.z, rlimits[0])
    ang_max = ang_min * rlimits[1] / rlimits[0]
    # compute pair counts
    if len(data_other) > 1:
        pairs = np.empty(len(data_reference))
        tree = SphericalKDTree(data_other.RA, data_other.DEC)
        for n, (row_idx, item) in enumerate(data_reference.iterrows()):
            # query the unknown tree between ang_min and ang_max
            idx, distance = tree.query_shell(
                item.RA, item.DEC, ang_min[n], ang_max[n])
            # compute pair count including optional weights
            if len(idx) > 0:
                weight = weights_other[idx]
                if inv_distance_weight:
                    pairs[n] = np.sum(weight / distance)
                else:
                    pairs[n] = np.sum(weight)
                if "weights" in item:  # reference weight
                    pairs[n] *= item.weights
            else:  # fallback for an empty slice
                pairs[n] = 0.0
    else:  # fallback for an empty group
        pairs = np.zeros(len(data_reference))
    # indices are needed to map the counts back to the correct reference object
    pair_counts = pd.DataFrame({"pairs": pairs}, index=data_reference.index)
    return pair_counts
