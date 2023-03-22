yet_another_wizz is a python package to efficiently compute cross-correlation
redshifts, also know as clustering redshifts and is hosted on github:

- code: https://github.com/jlvdb/yet_another_wizz.git
- docs: https://yet-another-wizz.readthedocs.io/

The method allows to estimate the unknown redshift distribution of a galaxy
sample by correlating the on-sky positions with a reference sample with known
redshifts. This implementation is based on the single bin correlation
measurement of the correlation amplitude, introduced by Schmidt et al. (2013,
`arXiv:1303.0292 <https://arxiv.org/abs/1303.0292>`_).

.. Note::
    When using this code in published work, please cite
    *van den Busch et al. (2020), A&A 642, A200*
    (`arXiv:2007.01846 <https://arxiv.org/abs/2007.01846>`_)


Features
--------

This code as been re-released as version 2.0, which includes a number of bug
fixes, new features, and performance and usability improvements:

- Measure the cross-correlation amplitude, including optional galaxy weights.
- Automatically mitigate galaxy bias by measuring the sample autocorrelation
  amplitudes (requires point redshift estimates).
- Enable empirical covariance estimation and paralellisation by using spatial
  regions.
- **New:** Count pairs across spatial region boundaries.
- **New:** Read from a varity of input file formats, such as FITS, Parquet,
  Feather, HDF5 and CSV.
- **New:** Supports for many correlation estimators (Landy-Szalay,
  Davis-Peebles, ...), including measuring the random-random pair count term.
- **New:** Apply random samples other than for the unknown sample, reference
  sample randoms (or both) are now supported for cross-correlation measurements.
- **New:** Performance and memory usage improvements.

.. Warning::
    Not all version 1 features are fully implemented. Missing features are:

    - Bootstrap resampling for covariance estimation (currently jackknifing).
    - Bias and redshift model fitting routines.


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

Example
^^^^^^^

The basic example below demonstrates how to measure clustering redshifts with
the ``yaw`` command line tool. For more details refer to the documentation or
use the builtin help invoked via ``yaw -h`` / ``yaw --help``.

1. Create a new project directory called ``output``, set the minimum
   configuration parameters and define reference sample input file::

    yaw init output \
        --rmin 100 --rmax 1000 \
        --zmin 0.07 --zmax 1.42 \
        --ref-path reference.fits \
        --ref-ra ra \
        --ref-dec dec \
        --ref-z z


2. Measure the cross-correlation pair counts and specify the unknown sample and
   random input file::

    yaw cross output \
        --unk-path unknown.fits \
        --unk-ra ra \
        --unk-dec dec \
        --rand-path random.fits \
        --rand-ra ra \
        --rand-dec dec

3. Compute correlation functions from the measured pair counts and create a
   simple check plot::

    yaw zcc output
    yaw plot output

That is all. The project directory should now contain a ``setup.yaml`` file with
all input parameters and tasks applied, a log file, and the
clustering redshifts stored as text file ``estimate/kpc100t1000/fid/nz_cc.dat``
together with a covariance estimate and jackknife samples.


Reporting bugs and requesting features
--------------------------------------

For bug reports or requesting new features, please use the github issue page:

https://github.com/jlvdb/yet_another_wizz/issues
