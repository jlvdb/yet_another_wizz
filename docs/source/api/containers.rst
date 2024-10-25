Data containers
===============

.. currentmodule:: yaw


The pair counts from the correlation measurements can be converted to a
correlation function and redshift estimte (including bias bias mitigation) with
the following classes. They store measurements, jackknife samples, and estimates
of the covariance matrix, as well as methods to save the data to files or
plotting them:

.. autosummary::
    :toctree: autogen

    CorrData
    RedshiftData


In case a data catalog has point redshifts attached, the following class can
be used to compute a redshift histogram with similar utility methods as those
above:

.. autosummary::
    :toctree: autogen

    HistData
