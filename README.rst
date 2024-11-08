..
    begin header

.. image:: https://raw.githubusercontent.com/jlvdb/yet_another_wizz/main/docs/source/_static/logo-dark.png
    :width: 1000
    :alt: yet_another_wizz

|

.. image:: https://img.shields.io/pypi/v/yet_another_wizz?logo=pypi&logoColor=blue
    :target: https://pypi.org/project/yet_another_wizz/
.. image:: https://github.com/jlvdb/yet_another_wizz/actions/workflows/docker-publish.yml/badge.svg
    :target: https://github.com/jlvdb/yet_another_wizz/actions/workflows/docker-publish.yml
.. image:: https://github.com/jlvdb/yet_another_wizz/actions/workflows/run-tests.yml/badge.svg
    :target: https://github.com/jlvdb/yet_another_wizz/actions/workflows/run-tests.yml
.. image:: https://readthedocs.org/projects/yet-another-wizz/badge/?version=latest
    :target: https://yet-another-wizz.readthedocs.io/en/latest/?badge=latest
.. image:: https://codecov.io/gh/jlvdb/yet_another_wizz/branch/main/graph/badge.svg?token=PC41ME2AR8
    :target: https://codecov.io/gh/jlvdb/yet_another_wizz

|

..
    end header

`yet_another_wizz` is a python package to efficiently compute cross-correlation
redshifts, also know as clustering redshifts. The method allows to estimate the
unknown redshift distribution of a galaxy sample by measuring the amplitude of
the angular correlation of the galaxy positions with those of a reference
sample with known redshifts.

This implementation is based on the idea (introduced by
Schmidt et al. 2013, see `arXiv:1303.0292 <https://arxiv.org/abs/1303.0292>`_)
to measure the amplitude of the angular correlation functions by counting galaxy
pairs in a single, wide angular bin.

The code base, documentation and, python package are distributed at:

- code: https://github.com/jlvdb/yet_another_wizz.git
- docs: https://yet-another-wizz.readthedocs.io/
- PyPI: https://pypi.org/project/yet_another_wizz/

.. note::
    In the latest version, the code has been redesigned for large data sets and
    now supports paralellism with MPI.


Installation
------------

The `yet_another_wizz` package, which ships the python library ``yaw``, can be
installed directly with `pip`::

    pip install yet_another_wizz

To enable MPI support, the MPI runtime-environment must be installed and
configured. The easiest way to install `yet_another_wizz` with MPI enabled is
using the provided setup for `conda`::

    conda install -f environment.yml

This will creates a new environment called ``yaw`` and install the code together
with the ``openmpi`` implementation of MPI.

Alternative use the pip install option::

    pip install yet_another_wizz[mpi]

Other optional dependencies (not installed by default) are:

- ``matplotlib`` to enable plotting methods.
- ``healpy`` to enable generating random samples based on `HealPix` masks.


Usage
-----

For more information about how to use the python code, please refer to the usage
examples in the official documentation.

There is also a `plugin interface <https://github.com/LSSTDESC/rail_yaw>`_ 
for the Redshift Assessment Infrastructure Layers
(`RAIL <https://github.com/LSSTDESC/rail>`_) pipeline.

Previous versions of `yet_another_wizz` could also be run as a command line tool
when installing the sparate command-line client `yet_another_wizz_cli`. This
tool is deprecated as of version 3.0 but maybe be integrated directly into
`yet_another_wizz` in a future release.

.. note::
    When using this code in published work, please cite
    *van den Busch et al. (2020), A&A 642, A200*
    (`arXiv:2007.01846 <https://arxiv.org/abs/2007.01846>`_)

For bug reports or requesting new features, please use the github issue page:
https://github.com/jlvdb/yet_another_wizz/issues


Maintainers
-----------

- Jan Luca van den Busch
  (*author*, Ruhr-Universität Bochum, Astronomisches Institut)


Acknowledgements
----------------

Jan Luca van den Busch acknowledges support from the European Research Council
under grant numbers 770935. The authors also thank Hendrik Hildebrandt,
Benjamin Joachimi, Angus H. Wright, and Chris Blake for vital feedback and
support throughout the development of this software.
