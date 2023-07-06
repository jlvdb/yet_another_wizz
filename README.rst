yet_another_wizz
================


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

This will install the python package ``yaw``, as well as the ``yaw`` executable
command line tool.


Usage
-----

There are two main ways to use yet_another_wizz,

- the ``yaw`` commmand line tool and
- the python package ``yaw`` directly.

Most people will probably get started with the command line tool, which should
cover all necessary tasks for a standard clustering redshift calibration. For
custom solutions, use the python package. A basic example as well as the API
reference can be found in the official documentation.


Reporting bugs and requesting features
--------------------------------------

For bug reports or requesting new features, please use the github issue page:

https://github.com/jlvdb/yet_another_wizz/issues

    
**Maintainers:**

- Jan Luca van den Busch
  (*author*, Ruhr-Universität Bochum, Astronomisches Institut)


Acknowledgements
----------------

Jan Luca van den Busch acknowledges support from the European Research Council
under grant numbers 770935. The authors also thank Hendrik Hildebrandt,
Benjamin Joachimi, Angus H. Wright, and Chris Blake for vital feedback and
support throughout the development of this software.
