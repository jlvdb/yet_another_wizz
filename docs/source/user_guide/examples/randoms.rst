Generating random points
------------------------

The code provides simple methods to :ref:`generate<generator>` uniform random
points. The simplest methods generates them within a rectangular footprint on
sky, i.e. in a fixed window of right ascension and declination. Additionally,
the method allows to draw samples from an array of observed redshifts or
weights, if desired. For example:

.. code-block:: python

    import yaw
    from yaw.randoms import BoxRandoms

    generator = BoxRandoms(
        ra_min=0.0,
        ra_max=90.0,
        dec_min=0.0,
        dec_max=90.0,
        # redshifts=None,
        # weights=None,
        # seed: int = 12345,
    )

    cat = yaw.Catalog.from_random(
        "path/to/cache",
        generator,
        num_randoms=10_000_000,
        # patch_centers=None,
        patch_num=64,
        # overwrite=False,
        progress=True,  # shows a progress bar, default: False
    )

To inspect the distribution, we need some additional code:

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt

    # need to iterate over patches and load coordinates manually
    coords = yaw.AngularCoordinates.from_coords(
        patch.coords for patch in cat.values()
    )
    plt.hist2d(coords.ra, np.sin(coords.dec), bins=90)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\sin{\delta}$")


.. figure:: /_static/rand_density.png
    :figwidth: 100%
    :alt: Example redshift estiamte

    Example distribution of randomly generated points from the example above in
    an equal-area projection.
