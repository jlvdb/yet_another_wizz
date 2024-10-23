.. _api:

API Documentation
=================

.. automodule:: yaw

.. Warning::
    Flowchart is outdated with missing UniformRandoms implementataion.

.. image:: /_static/flowchart.svg
    :width: 800
    :alt: API flowchart


.. rubric:: Catalog data containers

.. autosummary::
    :toctree: api

    Catalog
    Patch


.. rubric:: Correlation measurements

.. autosummary::
    :toctree: api

    Configuration
    config.ScalesConfig
    config.BinningConfig
    autocorrelate
    crosscorrelate


.. rubric:: Data containers

.. autosummary::
    :toctree: api

    paircounts.PatchedCounts
    paircounts.PatchedTotals
    paircounts.NormalisedCounts
    CorrFunc
    CorrData
    RedshiftData
    HistData


.. rubric:: Utilities

.. autosummary::
    :toctree: api

    AngularCoordinates
    AngularDistances
    Binning
