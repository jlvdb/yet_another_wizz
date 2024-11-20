.. _theexample:

Usage example
=============


This example illustrates how to estimte the redshift distribution of a catalog
with unknown redshifts. Additional inputs are a reference catalog with known
(typically) spectroscopic redshifts and a corresponding random catalog for the
reference sample:


.. dropdown:: :octicon:`list-ordered;1.5em` ‎ ‎ ‎ Example program
    :open:
    :color: muted
    :class-title: h5

    .. toctree::
        :glob:
        :maxdepth: 2

        examples/1*
        examples/2*
        examples/3*
        examples/4*


.. dropdown:: :octicon:`list-ordered;1.5em` ‎ ‎ ‎ Other useful snippets
    :open:
    :margin: 0
    :color: muted
    :class-title: h5

    .. toctree::
        :glob:
        :maxdepth: 2

        examples/randoms
        examples/logging


.. admonition:: Note for MPI users

    All the code in this example, including the I/O related methods, but except
    for the plotting, is compatible with MPI execution.

    To run this code in an MPI environment, save it as a script and launch it
    with your configured MPI exectuor, e.g.

    .. code-block:: sh

        mpiexec python yaw_script.py

.. caution::
    Previous versions of `yet_another_wizz` could also be run as a command line
    tool when installing the sparate command-line client ``yet_another_wizz_cli``.
    This tool is deprecated as of version 3.0 but maybe be integrated directly
    into `yet_another_wizz` in a future release.
