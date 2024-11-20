.. _api:

API Reference
=============

This API reference manual details the primary functions and object in the
`yet_another_wizz` python package ``yaw``.

.. figure:: /_static/flowchart.svg
    :figwidth: 80%
    :alt: API flowchart

    Example flowchart indicating the relationship between the most important
    public classes and functions in `yet_another_wizz` used for redshift
    estimation.


Catalog data
------------

.. currentmodule:: yaw

The following class manages input data catalogs with coordinates and optional
redshifts and weights:

.. autosummary::
    :toctree: autogen

    Catalog


Upon creating a new catalog, the input data is split into patches and cached
on disk. These patches are managed by the following classes:

.. autosummary::
    :toctree: autogen

    catalog.patch.Patch
    catalog.patch.Metadata


.. _generator:

Random generators
-----------------

.. currentmodule:: yaw

Catalogs can also be generated randomly, e.g. to create random samples for the
correlation measurements. The corresponding :meth:`~yaw.Catalog.from_random`
method accepts one of the following generators for uniform random data points:


.. autosummary::
    :toctree: autogen

    randoms.BoxRandoms
    randoms.HealPixRandoms


Configuration
-------------

.. currentmodule:: yaw

There is one central class that manages all configuration parameters for the
correlation measurements. These include the correlation scales, pair weighting,
and redshift binning for the reference catalog:

.. autosummary::
    :toctree: autogen

    Configuration


The configuration above is organised hierarchically. The main configuration
separates the scale and binning configuration into two sub-classes, however
usually it is not necessary to interact with these sub-classes directly.

.. autosummary::
    :toctree: autogen

    config.ScalesConfig
    config.BinningConfig


Correlation measurements
------------------------

.. currentmodule:: yaw

The cross- and autocorrelation measurements are implemented in two functions,
which both take a configuration object, and multiple data catalogs as input:

.. autosummary::
    :toctree: autogen

    crosscorrelate
    autocorrelate


Correlation Functions
---------------------

.. currentmodule:: yaw

The functions that measure correlations return a special container that stores
pair counts per redshift bin and spatial patches:

.. autosummary::
    :toctree: autogen

    CorrFunc


The pair counts stored in the class above are wrapped into another set of
containers that store the normalised pair counts per patch and redshift, which
in turn store the raw pair counts and the size of the patches (used to compute
the normalisation factor for the pair counts):

.. autosummary::
    :toctree: autogen

    correlation.NormalisedCounts
    correlation.PatchedCounts
    correlation.PatchedSumWeights


Data containers
---------------

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


Utilities
---------

.. currentmodule:: yaw

Most `yet_another_wizz` code produces detailed log messages. These can be
filtered and collected easily on standard output or in a log file using the
following utility function:

.. autosummary::
    :toctree: autogen

    utils.get_logger


There are two utility classes in `yet_another_wizz` that are used to handle
angular coordinates and distance computations in angular and 3-dimensional
Euclidean coordintes:

.. autosummary::
    :toctree: autogen

    AngularCoordinates
    AngularDistances


Additionally, there is a special container with convenience methods that store
bin intervals:

.. autosummary::
    :toctree: autogen

    Binning


Parameter options
-----------------

.. currentmodule:: yaw.options

The following string enumerations list commonly used configuration parameters
with a fixed set of options. They are usually interchangeable with string
values and are parsed automatically.

.. autosummary::
    :toctree: autogen
    :template: enum.rst

    BinMethodAuto
    BinMethod
    Closed
    CovKind
    PlotStyle
    Unit
