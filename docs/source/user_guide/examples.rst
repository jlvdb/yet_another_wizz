.. _theexample:

Usage example
=============


This example illustrates how to estimte the redshift distribution of a catalog
with unknown redshifts. Additional inputs are a reference catalog with known
(typically) spectroscopic redshifts and a corresponding random catalog for the
reference sample.

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

Initial setup
-------------

First, we import the `yet_another_wizz` python package:

.. code-block:: python

    import yaw

Next we configure the input file paths and directory in which we cache the
catalog data:

.. code-block:: python

    cache_dir = "path/to/fast/cache"  # create as needed
    reference_path = "path/to/ref_data.{fits,pqt,hdf5}"
    ref_rand_path = "path/to/rand_data.{fits,pqt,hdf5}"
    unknown_path = "path/to/unk_data.{fits,pqt,hdf5}"


Loading the input data
----------------------

First we need to create patch centers that we can use with all our input
catalogs. Since there are no pre-computed patch centers available, we generate
64 new patches when loading and caching the reference random catalog:

.. Note::
    Most optional function arguments are listed with their default values or as
    comments for convenience.

.. code-block:: python

    patch_num = 64

    cat_ref_rand = yaw.Catalog.from_file(
        cache_directory=f"{cache_dir}/ref_rand",
        path=ref_rand_path,
        ra_name="ra_column_name",
        dec_name="dec_column_name",
        weight_name="weight_column_name",  # optional
        redshift_name="zspec_column_name",  # required for reference
        # patch_centers=None,
        # patch_name=None,
        patch_num=patch_num,
        # degrees=True,
        # overwrite=False,
        progress=True,  # shows a progress bar, default: False
    )

    # extract the patch centers to use these for all following catalogs
    patch_centers = cat_ref_rand.get_centers()

In a similar way we load the other two data sets, but now use the patch centers
that we have generated above:

.. code-block:: python

    cat_reference = yaw.Catalog.from_file(
        cache_directory=f"{cache_dir}/reference",
        path=reference_path,
        ra_name="ra_column_name",
        dec_name="dec_column_name",
        weight_name="weight_column_name",  # optional
        redshift_name="zspec_column_name",  # required for reference
        patch_centers=patch_centers,  # use previously computed centers
        # patch_name=None,
        # patch_num=None,
        # degrees=True,
        # overwrite=False,
        progress=True,  # shows a progress bar, default: False
    )

    cat_unknown = yaw.Catalog.from_file(
        cache_directory=f"{cache_dir}/unknown",
        path=unknown_path,
        ra_name="ra_column_name",
        dec_name="dec_column_name",
        weight_name="weight_column_name",  # optional
        # we don't know the redshifts here, so we skip the argument
        patch_centers=patch_centers,  # use previously computed centers
        # patch_name=None,
        # patch_num=None,
        # degrees=True,
        # overwrite=False,
        progress=True,  # shows a progress bar, default: False
    )

    cat_unk_rand = None  # would be constructed same as cat_unknown


Measuring the correlations
--------------------------

First we set up the configuration for the correlation measurements. Here we want
to measure correlations corresponding to a transverse angular diameter distances
between 0.5 and 1.5 kpc. Additionally, we instruct the code to compute the
correlation functions in 22 linearly spaced bins of redshift in a range of
0.1 to 1.2:

.. code-block:: python

    config = yaw.Configuration.create(
        rmin=500.0,  # can also be a list of lower scale limits
        rmax=1500.0, # can also be a list of upper scale limits
        # rweight=None,     # if you want to weight pairs by scales
        # resolution=None,  # resolution of weights in no. of log-scale bins
        zmin=0.1,
        zmax=1.2,
        num_bins=22,
        # method="linear",
        # edges=None,  # provide your custom bin edges
    )

Next we measure the autocorrelation amplitude, which is a measure for the
galaxy bias of the reference sample. Afterwards we measure the cross-correlation
amplitude, which is the biased measure of the unknown redshift distribution.
Typically, this is the most expensive operation in the workflow:

.. code-block:: python

    cts_ss_list = yaw.autocorrelate(
        config,
        cat_reference,
        cat_ref_rand,
        progress=True,  # shows a progress bar, default: False
    )

    cts_sp_list = yaw.crosscorrelate(
        config,
        cat_reference,
        cat_unknown,
        ref_rand=cat_ref_rand,
        unk_rand=cat_unk_rand,
        progress=True,  # shows a progress bar, default: False
    )

The measurement functions above always return a list of correlation pair counts.
Since we configured a single measurement scale, the lists contain just a single
item, which are the pair counts that we are interested in. We can save them to
a HDF5 file for later inspection or avoiding to recompute the pair counts every
time.

.. code-block:: python

    cts_ss = cts_ss_list[0]
    cts_ss.to_file("w_ss.hdf5")

    cts_sp = cts_sp_list[0]
    cts_sp.to_file("w_sp.hdf5")
    # restored = yaw.CorrFunc.from_file("w_sp.hdf5")


Inspecting pair counts
~~~~~~~~~~~~~~~~~~~~~~

Correlation pair counts are stored as :obj:`yaw.CorrFunc` objects and are
very flexible. They can be sampled to an actual correlation function using a
correlation estimator,

.. code-block:: python

    w_ss = cts_ss.sample()  # creates a CorrFunc object
    w_ss.plot()  # automatic plot

or inspected (e.g. by indexing along the redshift bin or patch axis) to
investigate individual pair counts:

.. code-block:: python

    cts_sp.patches[3:6]  # subset with all pair counts involving patches 4 to 6
    cts_sp.bins[:5]  # subset with all pair counts of the first 5 redshift bins

    dd = cts_sp.dd  # access stored reference-unknown pair counts
    dd.get_array()  # array with shape (num_bins, num_patches, num_patches)


Computing the redshift estimate
-------------------------------

In the final step take the previously computed pair counts to transform them to
a redshift estimate. The code samples the correlation function and uses any
provided sample autocorrelation function as a bias correction term for the
measured cross-correlation:

.. code-block:: python

    ncc = RedshiftData.from_corrfuncs(
        cross_corr=cts_sp,
        ref_corr=cts_ss,
        # unk_corr=None,
    )

This special :obj:`~yaw.RedshiftData` object bundles the measured redshift
estimate, its uncertainty, jackknife samples, and a covariance matrix estimate:

.. code-block:: python

    ncc.data  # length num_bins
    ncc.error  # length num_bins
    ncc.samples  # shape (num_samples=num_patches, num_bins)
    ncc.covariance  # shape (num_bins, num_bins)

Similar to the pair counts, redshift estimates can be stored easily on disk,
however as three separate human-readable text files.

.. code-block:: python

    ncc.to_files("nz_estimate")
        # data/error         ->  nz_estimate.dat
        # jackknife samples  ->  nz_estimate.smp
        # covariance         ->  nz_estimate.cov
    # restored = yaw.RedshiftData.from_files("nz_estimate")

Additionally, the redshift estimate can be plotted easily:

.. code-block:: python
    
    ncc.plot(
        # label=None,
        # ax=None,  # plot to specific matplotlib axis
        # ...
    )

    # or even with estimated normalisation
    ncc.normalised().plot()


.. figure:: /_static/ncc_example.png
    :figwidth: 100%
    :alt: Example redshift estiamte

    Example for the automatic plot of the final redshift estimate obtained from
    small test samples.


Generating random points
------------------------

The code provides a simple method to generate uniform random points within a
rectangular footprint on sky, i.e. in a fixed window of right ascension and
declination. Additionally, the method allows to draw samples from an array
of redshifts or weights, if desired. For example:

.. code-block:: python

    from yaw.catalog import BoxGenerator

    generator = BoxGenerator(
        ra_min=0.0,
        ra_max=90.0,
        dec_min=0.0,
        dec_max=90.0,
        # redshifts=None,
        # weights=None,
        # seed: int = 12345,
    )

    cat = yaw.Catalog.from_random(
        "path/to/cache,
        generator,
        num_randoms=10_000_000,
        # patch_centers=None,
        patch_num=64,
        # overwrite=False,
        progress=True,  # shows a progress bar, default: False
    )
