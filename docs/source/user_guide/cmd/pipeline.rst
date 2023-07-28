.. _yaw_run:

Batch processing
----------------

One of the key features of ``yaw_cli`` is the ability to track all input
parameters and input files and to record all processing steps applied to the
data in a project. This information is stored in the ``setup.yaml`` file. This
serves two purposes:

- Reproduce the outputs from a single configuration file as long as the inputs
  are unchanged.
- Run ``yaw_cli`` in a batch-processing mode from a single configuration file
  instead of running multiple subcommands (``init``, ``cross``, ``zcc``, etc.)
  manually.


yaw_cli run
^^^^^^^^^^^

This batch-processing feature is implemented in the special subcommand
``yaw_cli run``. The command requires only two arguments, the name of the
output (project) directory, and the path to the ``setup.yaml`` file provided
with the ``--setup`` argument.

Optional arguments control the number of threads to use for parallel computing,
the verbosity level of the command line outputs, and providing a custom cache
directory location. More details can be obtained from the built-in help:

.. dropdown:: ``$ yaw_cli run --help``

    .. literalinclude:: yaw_help_run.txt
        :language: none


.. _conf_yaml:

Configuration file layout
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``setup.yaml`` configuration is a YAML file that has three main sections,
named `configuration`, `data` and `tasks`. A configuration file with default
values, place holders for file paths, and a list of all available tasks can be
generated as follows:

.. dropdown:: ``$ yaw_cli run --dump``
    :open:

    .. literalinclude:: default_setup.yaml
        :language: yaml

.. Note::

    All parameters with a leading ``(opt)`` in their comment are optional and
    can be omitted from the configuration file, the same applies to all items
    listed in `tasks`.


Configuration
"""""""""""""

This section maps one-to-one to a :class:`yaw.config.Configuration` instance and
specifies the correlation backend related parameters, the correlation
measurement scales, and the redshift binning. The parameter descriptions in the
box above are mostly self-explanatory, however there is one peculiarity:

.. Note::

    The configuration of the redshift bins has two mutually exclusive parameter
    group. The binning must specifed as either of:

    - ``binning.zbins``, i.e. providing a list of bin edges, or
    - ``binning.zmin``, ``binning.zmax``, (``binning.zbin_num``,
      ``binning.method``), i.e. providing parameters used to generate a binning
      automatically.

    If both are provided, ``binning.zbins`` is ignored.


Data
""""

This section specifes the input data files, split in two subsections `reference`
and `unknown`. Either section is optional, e.g. if no unknown sample is needed
for the tasks to perform, the section can be omitted.

Both sections each contain two subsections called `data` and `rand`, which
specify the data and optionally random datasets. While the `data` subsection is
always required, the `rand` can be omitted.

.. Note::

    Computing a crosscorrelation requires at least one of the two possible
    random samples (`data` or `rand`).

In each section, only the ``filepath``, ``ra``, and ``dec`` parameters are
required, the `reference` section additionally requries redshifts through the
``z`` parameter. In the `unknown` section, ``filepath`` may also specify many
input files (e.g. different tomographic bins), however these must all have the
same column names. Instead of providing a single file path, provide a mapping of
subset / bin index to file path, e.g.

.. code-block:: bash

    filepath:
        1: path/to/sample1
        5: path/to/sample5

instead of

.. code-block:: bash

    filepath: path/to/sample

.. Note::

    :ref:`patches`, which are used for error and covariance estimation, must be
    defined consistently for all input samples. Either use the ``n_patches``
    parameter to generate them automatically, or provide a column in the input
    files with an integer patch index using the ``patches`` parameters, e.g.::

        data:
            filepath: ...
            patches: name_of_patch_column


Tasks
"""""

This section is lists all tasks to be applied to the input data. The default
``setup.yaml`` will contain all possible tasks with a listing of all parameter
default values. The ``setup.yaml`` in a project directory always contains a
correctly ordered list of tasks (see above), without any duplicates (i.e.
replacing existing entries with the most recent calls).

Every task and all task parameters are optional and be omitted. For example,

.. code-block:: bash

    tasks:
        - cross:
              rr: false
        - zcc

and

.. code-block:: bash

    tasks:
        - cross
        - zcc

are equivalent, since ``rr: false`` is the default value. Note that the task
``zcc`` can be repeated arbitrarily many times, as long as the tag names differ.
If the tag name is identical, only the last version is kept. For example,

.. code-block:: yaml

    tasks:
        - cross
        - auto
        - zcc:
              tag: no_bias_mitigation
              bias_ref: false
        - zcc:
              tag: fid

will generate two redshift estimates, one called ``no_bias_mitigation``, which
does not use the reference sample autocorrelation to mitigate galaxy bias and
one called ``fid``, where the bias is mitigated. (Since ``fid`` is the default
tag, it is also possible to omit the last line entirely.)


Advanced usage
^^^^^^^^^^^^^^

The ``--config-from`` argument for ``yaw_cli run`` allows to rerun a previous
analysis setup (same input files and list of tasks), but using the
`configuration` section from a different input file. This is particularly useful
if one only wishes to change the measurement scales or redshift binning, etc.

For example

.. code-block:: bash

    yaw_cli version2 -s version1/setup.yaml --config-from new_config.yaml

creates a new project directory called ``version2``. The task list and input
files are taken from the setup file of an existing project called ``version1``,
but the configuration section is read from the ``new_config.yaml`` (ignoring
any other file contents).
