"""This module implements a class that generates the value options for certain
configuration parameters. An instance of this class can be accessed directly as
:obj:`yaw.config.OPTIONS`.
"""

from __future__ import annotations

__all__ = ["OPTIONS"]


class Options:
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


OPTIONS = Options()
