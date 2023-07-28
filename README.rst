.. image:: https://raw.githubusercontent.com/jlvdb/yet_another_wizz/main/docs/source/_static/logo-dark.png
    :width: 1000
    :alt: yet_another_wizz

|

.. image:: https://img.shields.io/pypi/v/yet_another_wizz?logo=pypi&logoColor=blue
    :target: https://pypi.org/project/yet_another_wizz/
.. image:: https://readthedocs.org/projects/yet-another-wizz/badge/?version=latest
    :target: https://yet-another-wizz.readthedocs.io/en/latest/?badge=latest
.. image:: https://github.com/jlvdb/yet_another_wizz/actions/workflows/run-tests.yml/badge.svg
    :target: https://github.com/jlvdb/yet_another_wizz/actions/workflows/run-tests.yml
.. image:: https://codecov.io/gh/jlvdb/yet_another_wizz/branch/main/graph/badge.svg?token=PC41ME2AR8
    :target: https://codecov.io/gh/jlvdb/yet_another_wizz
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/

|

*yet_another_wizz* is a python package to efficiently compute cross-correlation
redshifts, also know as clustering redshifts and is hosted on github:

- code: https://github.com/jlvdb/yet_another_wizz.git
- docs: https://yet-another-wizz.readthedocs.io/
- PyPI: https://pypi.org/project/yet_another_wizz/

The method allows to estimate the unknown redshift distribution of a galaxy
sample by correlating the on-sky positions with a reference sample with known
redshifts. This implementation is based on the single bin correlation
measurement of the correlation amplitude, introduced by Schmidt et al. (2013,
`arXiv:1303.0292 <https://arxiv.org/abs/1303.0292>`_).

.. Note::
    When using this code in published work, please cite
    *van den Busch et al. (2020), A&A 642, A200*
    (`arXiv:2007.01846 <https://arxiv.org/abs/2007.01846>`_)


Installation
------------

The yet_another_wizz package can be installed directly with pip::

    pip install yet_another_wizz

This will install the python library ``yaw``.

Commandline tool
~~~~~~~~~~~~~~~~

There also exists a separate command line tool called
`yet_another_wizz_cli <https://github.com/jlvdb/yet_another_wizz_cli>`_
(``yaw_cli``) that is available at PyPI and github. To install it alongside the
python library, type::

    pip install yet_another_wizz[cli]

LSST-DESC RAIL plugin
~~~~~~~~~~~~~~~~~~~~~

Currently there is also a
`plugin interface <https://github.com/jlvdb/yet_another_wizz_rail>`_  for the
Redshift Assessment Infrastructure Layers
(`RAIL <https://github.com/LSSTDESC/rail>`_) pipeline under development. To
install it alongside the python library, including ``rail`` itself, type::

    pip install yet_another_wizz[rail]


Usage
-----

There are two main ways to use yet_another_wizz,

- the python library ``yaw`` itself and
- the (separate) ``yaw_cli`` commmand line tool.
- the (separate) ``yaw_rail`` RAIL plugin (coming soon).

Most users will probably get started with the command line tool, which should
cover all necessary tasks for a standard clustering redshift calibration. For
custom solutions, use the python library. A basic example as well as the API
reference can be found in the official documentation.


Reporting bugs and requesting features
--------------------------------------

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
