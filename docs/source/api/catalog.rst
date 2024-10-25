Catalog data
============

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

    catalog.Patch
    catalog.Metadata
