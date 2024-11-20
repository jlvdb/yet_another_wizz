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
        # unit="kpc"  # defaults to angular diameter distance, but angles and
                      # comoving transverse distance are supported
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
