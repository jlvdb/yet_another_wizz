.. _yaw_init:

Starting a new project
----------------------

The command line tool is designed to operate on separate projects. A project
uses a fixed set of parameters (e.g. redshift binning and correlation
measurement scales) to compute clustering redshifts with a single reference
sample and one or many unknown data sets that spatially overlap.

.. Note::

    Using multiplereference samples as input for a project is currently not
    supported, however it is possible to :ref:`merge<yaw_merge>` the outputs
    from different projects.

New projects are created with the ``yaw_cli init [path]`` subcommand, where the
``path`` specifies a directory (must not exist) in which all data products are
:ref:`stored and managed<projdir>`. This command specifies the majority of the
paramters for the correlation measurements, including the measurement scales,
the redshift binning, as well as optional parameters such as the cosmological
model for distance calculations and the automatic generation of
:ref:`spatial patches<patches>`. A list of all command line arguments can be
obtained by typing

.. dropdown:: ``$ yaw_cli init --help``

    .. literalinclude:: yaw_help_init.txt
        :language: none

.. Note::

    The configuration of the redshift bins has two mutually exclusive parameter
    group. The binning must specifed as either of:

    - ``--zbins``, i.e. providing a list of bin edges, or
    - ``--zmin``, ``--zmax``, (``--zbin-num``, ``--method``), i.e. providing
      parameters used to generate a binning automatically.

    If both are provided, ``--zbins`` is ignored.


The reference sample
^^^^^^^^^^^^^^^^^^^^

Since the reference sample used for a project is static, the reference sample is
already specifed at this stage by providing an input path ``--ref-path`` and the
requred column names for right ascension (``--ref-ra``), declination
(``--ref-dec``, in degrees) and per-object redshifts (``--ref-z``), weights
(``--ref-w``) are optional.

Similarly, a random sample for the reference sample can be provided using the
corresponding ``--rand-*`` arguments. Note that the reference randoms also
require per-object redshifts. If no reference randoms are provided, randoms for
the unknown sample are required (see :ref:`yaw_cross`).


Spatial patches and caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is important to specify consistent spatial patches for a project, since these
are used to compute uncertainty estimates and covariances. There are two
options:

1. Generate the patches automatically using a k-means clustering algorithm. The
   code ensures that all data and random catalogues have the patch centers.
2. Provide manual patch assignements from a column with integer patch indices
   ``--ref-patch`` and ``--rand-patch``. The code will only check that the
   patches align roughly, but the user must ensure that they are consistent for
   all input samples.

.. Warning::

    For performance reasons it is highly recommended to cache all input data
    sets using the flags ``--ref-cache`` and ``--rand-cache``. For more details
    refer to :ref:`caching`.


Outputs
^^^^^^^

The ``init`` subcommand creates an empty :ref:`project directory<projdir>`, in
which all data products are stored. The configuration is stored in the newly
created ``setup.yaml`` :ref:`YAML file<conf_yaml>`, together with a declaration
of input files and processing steps applied (see next page). Logs for debugging
are stored in ``setup.log``, the patch center coordinates are stored in
``patch_centers.dat``. Finally, the redshift distribution of the reference
sample is computed and stored as ``true/nz_reference.*``.
