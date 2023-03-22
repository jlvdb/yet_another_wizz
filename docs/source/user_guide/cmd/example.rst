.. _quickcmd:

Quick start
-----------

We start by providing a *minimal working example* for a very basic clustering
redshift analysis with additional correction of the reference sample bias. You
can find the same example using the python API :ref:`here<quickapi>`.

This example assumes that the input catalogs are provided as FITS data files,
one for the reference sample (``reference.fits``) and a matching random catalog
(``randoms.fits``), and one for the unknown sample (``unknown.fits``). These
files contain right ascension and declination in degrees, named ``ra`` and
``dec``, as well as optional redshifts ``z``.


Creating a new project
^^^^^^^^^^^^^^^^^^^^^^

The command line tools operate on a single, unified output directory, in which
configuration, input and output data are organised automatically. We start by
creatint a new project called ``output`` with the ``yaw init`` command and set
the minimum required configuration parameters. We also define the input
reference sample and random catalog by listing the required column names.

Finally we want to split the data catalogs into ``32`` spatial patches that
allow us later to get uncertainties for our clustering redshift measurements.
By default these are created automatically using a k-means clustering algorithm.

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

We cannot change the reference sample since every project must have a unique
reference sample.


Measuring correlations
^^^^^^^^^^^^^^^^^^^^^^

Next we want to measure the crosscorrelation of the reference sample with the
unknown catalog, which we specify when running the ``yaw cross`` command.
Note that we can in principle provide as many input files with ``--unk-path``
as we would like (e.g. tomographic bins).

In the same way we measure the autocorrelation function of the reference sample
to mitigate its galaxy bias evolution. In our case, the ``yaw auto`` command
takes no further inputs since most run parameters are already configured at this
point.

.. code-block:: console

    yaw cross output \
        --unk-path unknown.fits \
        --unk-ra ra \
        --unk-dec dec
    yaw auto output

Note that these two tools only measure the correlation pair counts. Here the
cross-correlation contains the data-data and data-random counts, whereas the
autocorrelation by defaults also includes the random-random pair counts.


Getting the clustering redshifts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally we transform the pair count into correlation functions and obtain the
clustering redshift estimate with ``yaw zcc``. We also create a simple check
plot with the ``yaw plot`` tool.

.. code-block:: console

    yaw zcc output
    yaw plot output

That is all. The project directory should now contain a ``setup.yaml`` file with
all input parameters and tasks applied, a log file, and the
clustering redshifts stored as text file ``estimate/kpc100t1000/fid/nz_cc.dat``
together with a covariance estimate and jackknife samples.


Batch processing
----------------

The script calls above can be summarised into a single configuration file, which
is generated automatically as we run the scripts. These setup YAML files can
be freely configured and run with the batch processing tool ``yaw run``.

We can produce the same clustering redshfit estimate by running

.. code-block:: console

    yaw run output -s setup.yaml

where the contents of ``setup.yaml`` are as follows:

.. literalinclude:: example.yaml
    :language: yaml
