"""This module implements a class that generates the value options for certain
configuration parameters. An instance of this class can be accessed directly as
:obj:`yaw.config.OPTIONS`.
"""

from __future__ import annotations

__all__ = ["OPTIONS"]


class Options:
    @property
    def backend(self) -> tuple[str]:
        """Lists the names of the currently available backends for correlation
        measurements."""
        from yaw.catalogs import BaseCatalog

        return tuple(sorted(BaseCatalog._backends.keys()))

    @property
    def binning(self) -> tuple[str]:
        """Lists the currently implemented methods to generate redshift bins
        with the :obj:`~yaw.config.BinningConfig` class.

        .. rubric:: Values

        ``comoving``: Generate a binning with equal width in radial comoving
        distance.

        ``linear``: Generate a binning with equal width in redshift.

        ``logspace``: Generate a binning with equal width in logarithmic
        redshift :math:`\\log(1+z)`.

        .. Note::

            Class also accepts ``manual`` if custom bin edges are provided.
        """
        return ("comoving", "linear", "logspace")

    @property
    def cosmology(self) -> tuple[str]:
        """Lists the availble named cosmologies in :obj:`astropy`.

        On top of these comological models, custom cosmologies can be defined by
        subclassing :obj:`yaw.core.cosmology.CustomCosmology`.
        """
        from astropy.cosmology import available

        return available

    @property
    def kind(self) -> tuple[str]:
        """Lists the currently implemented methods for covariance calculation.

        .. rubric:: Values

        ``full``: Compute all matrix elements of the covariance.

        ``diag``: Compute only the main diagonal and the primary off-diagonals
        of the covariance matrix. This option applies, if the covariance is
        computed from a concatenated set of data samples, which have a
        crosscorrelation of interest. For example if concatenating the samples
        obtained from multiple redshift bins, the primary off-diagonals contain
        the covariance at the same redshift between different bins.

        ``var``: Compute the variance, i.e. only the diagonal elements.
        """
        return ("full", "diag", "var")

    @property
    def merge(self) -> tuple[str]:
        """Lists the available modes to merge correlation measurements.

        .. rubric:: Values

        ``patches``: Merge measurements by concatenating spatial patches, i.e.
        extending the area over which measurements are taken. This may miss out
        some correlation signal between the two measurements.

        ``redshift``: Merge measurements by concatenating redshift bins, i.e.
        extending the redshift range.
        """
        return ("patches", "redshift")

    @property
    def method(self) -> tuple[str]:
        """Lists the currently implemented methods for spatial resampling.

        Resampling uses the spatial patches to get uncertainty estimates for the
        correlation function measurements.

        .. rubric:: Values

        ``jackknife``: Use jackknife resampling (generate samples and leave out
        one patch at a time).

        ``bootstrap``: Use bootstrap resampling (generate samples by randomly
        drawing patches with replacement).
        """
        return ("jackknife", "bootstrap")


OPTIONS = Options()
