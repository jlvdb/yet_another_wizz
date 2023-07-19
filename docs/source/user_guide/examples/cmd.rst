.. _quickcmd:

Using the command tools
-----------------------

The command line tool ``yaw_cli``
(`separate installation required <https://github.com/jlvdb/yet_another_wizz_cli>`_)
operate on a single, unified :ref:`output directory<projdir>`, in which
configuration, input and output data are organised automatically.


Creating a new project
^^^^^^^^^^^^^^^^^^^^^^

We start by creating a new project called ``output`` with the ``yaw_cli init``
:ref:`command<yaw_init>` and set the minimum required configuration parameters.
We also define the input reference sample and random catalog and list the
required column names.

Finally we want to split the data catalogs into ``32``
:ref:`spatial patches<patches>` that allow us later to get uncertainties for our
clustering redshift measurements. By default these are created automatically
using a k-means clustering algorithm.

.. code-block:: bash

    $ yaw_cli init output \
        --rmin 100 --rmax 1000 \
        --zmin 0.07 --zmax 1.42 \
        --ref-path reference.fits \
        --ref-ra ra \
        --ref-dec dec \
        --ref-z z \
        --rand-path random.fits \
        --rand-ra ra \
        --rand-dec dec \
        --rand-z z \
        --n-patches 32

.. Note::

    Every project uses a unique reference sample. We cannot change the reference
    sample after creating the project.


Measuring correlations
^^^^^^^^^^^^^^^^^^^^^^

Next we want to measure the crosscorrelation of the reference sample with the
unknown catalog, which we specify when running the ``yaw_cli cross``
:ref:`command<yaw_cross>`. Note that we can in principle provide as many input
files with ``--unk-path`` as we would like (e.g. tomographic bins).

In the same way we measure the autocorrelation function of the reference sample
to mitigate its galaxy bias evolution. In our case, the ``yaw_cli auto``
:ref:`command<yaw_auto>` takes no further inputs since most run parameters,
including the reference sample, are already configured at this point.

.. code-block:: bash

    $ yaw_cli cross output \
        --unk-path unknown.fits \
        --unk-ra ra \
        --unk-dec dec
    $ yaw_cli auto output

.. Note::

    These two tools only measure the correlation pair counts. Here the
    cross-correlation contains the data-data and data-random counts, whereas the
    autocorrelation by defaults also includes the random-random pair counts.


.. _projoutputs:

Getting the clustering redshifts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally we transform the pair count into correlation functions and obtain the
clustering redshift estimate with the ``yaw_cli zcc`` :ref:`command<yaw_zcc>`.
We also create a simple check plot with the
``yaw_cli plot`` :ref:`command<yaw_plot>`.

.. code-block:: bash

    $ yaw_cli zcc output
    $ yaw_cli plot output

That is all. The project directory should now contain a number of files, the
most important ones are:

.. code-block::

    output/
    ├─ estimate/
    │  ├─ kpc100t1000/
    │  │  └─ fid/
    │  │     ├─ auto_reference.dat
    │  │     └─ nz_cc_1.dat
    │  ├─ auto_reference.png
    │  └─ nz_estimate.png
    ├─ setup.log
    └─ setup.yaml

The first file is a YAML configuration file which records all configuration,
inputs and tasks applied, which :ref:`makes this run reproducable<yaw_run>`.

The ``estimate`` directory contains the check plots of the redshift estimate and
the reference sample autocorrelation function, which is a proxy for the galaxy
bias. The data products are stored in ``kpc100t100/fid``, the default name for
our choice of scales. They are named ``n_cc_1.dat`` (redshifts estimate) and
``auto_reference.dat`` (reference autocorrelation) and are accompanied by a
covariance matrix and jackknife samples in separate files.

Finally, there are automatically generated checkplots in the ``estimate``
directory, one for the reference sample autocorrelation function and one for
the redshift estimate.


Tomographic binning and other subsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the unknown sample is split into different subsets, e.g. tomographic redshift
bins, these can be processed easily with ``yaw_cli`` by providing a list of
unknown (and optionally random) data catalogues, e.g.:

.. code-block:: bash

    $ yaw_cli cross output \
        --unk-path unknown1.fits unknown2.fits unknown3.fits \
        --unk-ra ra \
        --unk-dec dec

This would produce clustering redshift estimates for three subsets of the
unknown data, in each case using the same reference sample as before. The
redshift estimates in ``estimate/kpc100t100/fid`` are numbered automatically
(counting from 1) and are called  ``n_cc_1.dat``, ``n_cc_2.dat``, and
``n_cc_3.dat`` for this example. The automatically generated checkplot will
contain three panels instead of one.
