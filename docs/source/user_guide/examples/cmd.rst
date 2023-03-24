.. _quickcmd:

Using the command tools
-----------------------

The command line tools operate on a single, unified
:ref:`output directory<projdir>`, in which configuration, input and output data
are organised automatically.


Creating a new project
^^^^^^^^^^^^^^^^^^^^^^

We start by creating a new project called ``output`` with the ``yaw init``
:ref:`command<yaw_init>` and set the minimum required configuration parameters.
We also define the input reference sample and random catalog and list the
required column names.

Finally we want to split the data catalogs into ``32``
:ref:`spatial patches<patches>` that allow us later to get uncertainties for our
clustering redshift measurements. By default these are created automatically
using a k-means clustering algorithm.

.. code-block:: console

    yaw init output \
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
unknown catalog, which we specify when running the ``yaw cross``
:ref:`command<yaw_cross>`. Note that we can in principle provide as many input
files with ``--unk-path`` as we would like (e.g. tomographic bins).

In the same way we measure the autocorrelation function of the reference sample
to mitigate its galaxy bias evolution. In our case, the ``yaw auto``
:ref:`command<yaw_auto>` takes no further inputs since most run parameters are
already configured at this point.

.. code-block:: console

    yaw cross output \
        --unk-path unknown.fits \
        --unk-ra ra \
        --unk-dec dec
    yaw auto output

.. Note::

    These two tools only measure the correlation pair counts. Here the
    cross-correlation contains the data-data and data-random counts, whereas the
    autocorrelation by defaults also includes the random-random pair counts.


.. _projoutputs:

Getting the clustering redshifts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally we transform the pair count into correlation functions and obtain the
clustering redshift estimate with the ``yaw zcc`` :ref:`command<yaw_zcc>`. We
also create a simple check plot with the ``yaw plot`` :ref:`command<yaw_plot>`.

.. code-block:: console

    yaw zcc output
    yaw plot output

That is all. The project directory should now contain a number of files, the
most important ones are:

.. code-block::

    output/
    ├─ estimate/
    │  ├─ kpc100t1000/
    │  │  └─ fid/
    │  │     ├─ auto_reference.dat
    │  │     └─ nz_cc_0.dat
    │  ├─ auto_reference.png
    │  └─ nz_estimate.png
    ├─ setup.log
    └─ setup.yaml

The first file is a YAML configuration file which records all configuration,
inputs and tasks applied, which :ref:`makes this run reproducable<yaw_run>`.

The ``estimate`` directory contains the check plots of the redshift estimate and 
the reference sample autocorrelation function, which is a proxy for the galaxy
bias. The data products are stored in ``kpc100t100/fid``, the default name for
our choice of scales. They are named ``n_cc_0.dat`` (redshifts estimate) and
``auto_reference.dat`` (reference autocorrelation) and are accompanied by a
covariance matrix and jackknife samples in separate files.
