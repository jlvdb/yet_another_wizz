Measurement configuration
=========================


*yet_another_wizz* uses configuration objects to organise its free parameters.
These are implemented in the :mod:`~yaw.config` module, the most important one,
:class:`~yaw.config.Configuration`, is directly availale after importing the
main module.

.. code-block:: python

    from yaw import Configuration

The recommended way to create a new configuration is through its constructor
method, which is a keyword-only function. A minimum example is:

.. code-block:: python

    >>> config = Configuration.create(
    ...     rmin=100, rmax=1000, zmin=0.07, zmax=1.42)

The configuration object is hierarchical and holds the parameters in its
attributes. The structure matches the layout of the YAML configuration file used
for :ref:`batch processing<yaw_run>`:

.. code-block:: python

    >>> config.scales
    ScalesConfig(rmin=100.0, rmax=1000.0, rweight=None, rbin_num=50)
    >>> config.binning
    AutoBinningConfig(zbin_num=30, z='0.070...1.420', method='linear')
    >>> config.backend
    BackendConfig(thread_num=8, crosspatch=True, rbin_slop=0.01)
    >>> config.cosmology
    FlatLambdaCDM(name="Planck15", H0=67.74 km / (Mpc s), Om0=0.3075, Tcmb0=2.7255 K, Neff=3.046, m_nu=[0.   0.   0.06] eV, Ob0=0.0486)

The sections ``scales``, ``binning``, ``backend``, and ``cosmology`` are
configuration objects themselves, that are constructed automatically (see
below). The attributes of these sections can be accessed through the section
itself, e.g.:

.. code-block:: python

    >>> config.scales.rmin
    100.0
    >>> config.binning.zmax
    1.42


.. Note::

    To protect configuration objects from accidental modification, all
    attributes are read-only. To modify a configuration, use the
    :meth:`~yaw.config.Configuration.modify` method, which supports the same
    arguments as :meth:`~yaw.config.Configuration.create`. The method creates
    a new instance and replaces all values by the given keyword arguments, e.g.:

    .. code-block:: python

        >>> config.scales.rmin, config.scales.rmin
        (100.0, 1000.0)
        
        >>> new = config.modify(rmin=200, rmax=2000)
        >>> new.scales.rmin, new.scales.rmax
        (200.0, 2000.0)


``Configuration.scales``
------------------------

Scales for correlation measurements are implemented in the
:class:`~yaw.config.ScalesConfig` class and provided as projected physical
distance in kpc. The lower and upper limits, ``rmin`` and ``rmax``, can either
be scalar or arrays of the same length to specify more than one scale.

Optional parameter are ``rweight`` and ``rbin_num``, which are used to apply a
radial weighting to the single-bin correlation measurement. If ``rweight`` is
provided, the correlation function is measured in ``rbin_num`` logarithmically
spaced radial bins. The pair counts are multiplied by the radius corresponding
to the bin center and the summed together to approximate a radially weighted
correlation measurement.


``Configuration.binning``
-------------------------

The redshift binning is either constructed automatically using the ``zmin`` and
``zmax`` (and optional ``zbin_num`` and ``method``) parameters, or assigned
manually using the ``zbins`` parameter:

.. code-block:: python

    >>> Configuration.create(
    ...     rmin=100, rmax=1000, zbins=[0.1, 0.2, 0.3, 0.4])

    >>> Configuration.create(
    ...     rmin=100, rmax=1000, zmin=0.07, zmax=1.42,
    ...     zbin_num=30, method="linear")

The former creates a :class:`~yaw.config.ManualBinningConfig` object, the latter
constructs a :class:`~yaw.config.AutoBinningConfig` class. The case shown above
corresponds to the default binning, which are 30 bins, linearly spaced in
redshift. Other spacings can be selected using the method parameter, see also
:const:`~yaw.cosmology.BINNING_OPTIONS`.

.. Note::

    Either ``zmin`` and ``zmax`` or ``zbins`` are required to construct a valid
    redshift binning, otherwise a :exc:`~yaw.config.ConfigurationError` is
    raised.


``Configuration.backend``
-------------------------

This section maps to the :class:`~yaw.config.BackendConfig` class, which are
parameters for the backend used to compute correlations (see
:const:`~yaw.catalogs.BACKEND_OPTIONS` and the section on
:ref:`data catalogs<api_catalogs>`). The most important parameter here is the
``thread_num`` parameter, which specifies the number of parallel threads to use.
The ``crosspatch`` parameter specifies, whether the backend counts pairs beyond
patch boundaries (``crosspatch=False`` not supported by all backends).


``Configuration.cosmology``
---------------------------

The cosmological model that is used for distance calculation has usually a minor
effect on clustring redshifts. In the configuration it is specifed through the
``cosmology`` parameter, the current default is ``cosmology=Planck15``.

If you need use a model, a number of
:const:`named models<yaw.cosmology.COSMOLOGY_OPTIONS>` from the
:mod:`astropy.cosmology` module are available. If you need a custom cosmological
model, make sure to implemented it as subclass of
:class:`yaw.cosmology.CustomCosmology` and overwrite the methods that
*yet_another_wizz* expects, e.g.:

.. code-block:: python

    class MyCosmology(CustomCosmology):

        def comoving_distance(self, z: ArrayLike) -> ArrayLike:
            return my_comoving_distance(z)

        def comoving_transverse_distance(self, z: ArrayLike) -> ArrayLike:
            return my_comoving_transverse_distance(z)


