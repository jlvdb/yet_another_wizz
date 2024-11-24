Redshift estimates
------------------

.. currentmodule:: yaw

The pair counts from the :ref:`correlation measurements<corrfunc>` can be
sampled and converted to a correlation function, measured in bins of redshifts.
This can be achieved with the following class, which store measurements,
jackknife samples, and estimates of the covariance matrix, as well as methods to
save the data to files or plotting them:

.. autosummary::
    :toctree: autogen

    CorrData


The final redshift estimate (including optional bias mitigation) can be computed
from a cross correlation function measurement (and optional autocorrelation
measurements) with the following container. It provides similar functionality as
the :obj:`CorrData` above:

.. autosummary::
    :toctree: autogen

    RedshiftData


In case a data catalog has point redshifts attached, the following class can
be used to compute a redshift histogram with similar utility methods as those
above:

.. autosummary::
    :toctree: autogen

    HistData
