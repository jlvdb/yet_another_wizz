Command line tools
==================

The *yet_another_wizz* package can be used with a command line tool called
``yaw_cli``. This tool is no longer shipped with the ``yaw`` python library, but
requires a (`separate installation <https://github.com/jlvdb/yet_another_wizz_cli>`_)::

    pip install yet_another_wizz_cli


Features
--------

The aim of this command line tool is to cover and automate the most common use
cases of computing clustering redshifts and the mitigation of galaxy bias. A few
advantages are:

- **Automatic data management** in :ref:`project directories<projdir>`.
- **Automatic summary** of the input data and parameters, including a list of
  applied tasks. This allows to **reproduce the results** as long as the
  original input files are available.
- **Event logging** for debugging purposes.
- **Batch computing** of clustering redshifts from a single configuration file.


Usage
-----

A first overview of its features can be obtained from the built-in help tool of
the main command ``yaw_cli``. This generates a listing of all subcommands with a
brief summary of their purpose:

.. dropdown:: ``$ yaw_cli --help``

    .. literalinclude:: cmd/yaw_help_.txt
        :language: none

The details of these subcommands are described
in the following sections.


.. toctree::
    :hidden:

    cmd/init
    cmd/scripts
    cmd/pipeline
    cmd/projdir
    cmd/merging
