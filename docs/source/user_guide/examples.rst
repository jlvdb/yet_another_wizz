.. _theexample:

Usage example
=============


.. code-block:: python

    import yaw


.. code-block:: python

    PROGRESS = True  # if you want to see a progress bar

    patch_num = 64  # code will generate this number of patch centers from the reference randoms
    config = Configuration.create(
        rmin=500.0,  # scalar or list of lower scale cuts
        rmax=1500.0, # scalar or list of upper scale cuts
        zmin=0.1,
        zmax=1.2,
        num_bins=22,
    )


.. code-block:: python

    CACHE_DIR = "......"
    delete_and_recreate_cache_directory(CACHE_DIR)

    cat_ref_rand = Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "ref_rand"),
        path="path/to/ref_rand.{fits,pqt,hdf5}",
        ra_name="ra_column",
        dec_name="dec_column",
        redshift_name="zspec_column",
        weight_name="weight_column",  # optional
        patch_num=patch_num,
        progress=PROGRESS,
    )
    patch_centers = cat_ref_rand.get_centers()  # use these for all following catalogs

    cat_reference = Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "reference"),
        path="path/to/reference.{fits,pqt,hdf5}",
        ra_name="ra_column",
        dec_name="dec_column",
        redshift_name="zspec_column",
        weight_name="weight_column",  # optional
        patch_centers=patch_centers,
        progress=PROGRESS,
    )

    cat_unknown = Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unknown"),
        path="path/to/unknown.{fits,pqt,hdf5}",
        ra_name="ra_column",
        dec_name="dec_column",
        weight_name="weight_column",  # optional
        patch_centers=patch_centers,
        progress=PROGRESS,
    )

    cat_unk_rand = None  # assuming you don't have this


.. code-block:: python

    w_ss = autocorrelate(
        config,
        cat_reference,
        cat_ref_rand,
        progress=PROGRESS
    )[0]  # returns a list, one for each scale, just pick the first here
    #   w_ss.to_file("...") -> store correlation pair counts as HDF5 file

    w_sp = crosscorrelate(
        config,
        cat_reference,
        cat_unknown,
        ref_rand=cat_ref_rand,
        unk_rand=cat_unk_rand,
        progress=PROGRESS
    )[0]  # returns a list, one for each scale, just pick the first here
    #   w_sp.to_file("...") -> store correlation pair counts as HDF5 file

    # if you have mock data + unk_rand and know the true redshifts you can also do this:
    #   w_pp = autocorrelate(config, cat_unknown, cat_unk_rand, progress=True)[0]

.. code-block:: python

    ncc = RedshiftData.from_corrfuncs(cross_corr=w_sp, ref_corr=w_ss)  # unk_corr=w_pp
    ncc.to_files("nz_estimate")  # store as ASCII files with extensions .dat, .smp and .cov
    # useful attributes:
    #   ncc.data -> estiamte
    #   ncc.error -> Gaussian error estimate
    #   ncc.covariance -> jackknife covariance
    #   ncc.plot()
