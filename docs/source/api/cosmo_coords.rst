Coordinates & Cosmology
-----------------------

.. currentmodule:: yaw

There are two utility classes in `yet_another_wizz` that are used to handle
angular coordinates and distance computations in angular and 3-dimensional
Euclidean coordintes:

.. autosummary::
    :toctree: autogen

    AngularCoordinates
    AngularDistances


In many cases, the code needs to convert physical or comoving distances to
angles at a given redshift. Although the final clustering redshifts should only
very weakly depend on the exact cosmological model, it is possible to choose the
cosmology freely. Valid cosmologies must be either `astropy` cosmologies or a
subclass of :obj:`~yaw.cosmology.CustomCosmology`:

.. autosummary::
    :toctree: autogen

    cosmology.CustomCosmology
    cosmology.get_default_cosmology
