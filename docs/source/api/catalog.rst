Catalog and random data
-----------------------

The primary input for `yet_another_wizz` are tabular datasets which are managed
by a special :obj:`~yaw.Catalog` class. Catalogs can be created from in-memory
datasets, tabular data files, and random generators.

.. _catalogs:

The Catalog class
~~~~~~~~~~~~~~~~~

.. currentmodule:: yaw

A catalog is a collection of patches of catalog data (coordinates, weights,
redshifts, etc.), which are stored in a cache directory on disk:

.. autosummary::
    :toctree: autogen

    Catalog


Upon creating a new catalog, the input data is split into patches which are
stored separately in the catalog's cache directory. A single patch and its
metadata is managed by the following classes:

.. autosummary::
    :toctree: autogen

    catalog.patch.Patch
    catalog.patch.Metadata


.. _generator:

Random generators
~~~~~~~~~~~~~~~~~

.. currentmodule:: yaw

Catalogs can also be generated randomly, e.g. to create random samples for the
correlation measurements. The corresponding :meth:`~yaw.Catalog.from_random`
method accepts one of the following generators for uniform random data points:


.. autosummary::
    :toctree: autogen

    randoms.BoxRandoms
    randoms.HealPixRandoms
