.. _generator:

Random generators
=================

.. currentmodule:: yaw


Catalogs can also be generated randomly, e.g. to create random samples for the
correlation measurements. The corresponding :meth:`~yaw.Catalog.from_random`
method accepts one of the following generators for uniform random data points:


.. autosummary::
    :toctree: autogen

    randoms.BoxRandoms
    randoms.HealPixRandoms
