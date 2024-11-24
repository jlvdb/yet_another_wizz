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
