.. _patches:

Spatial patches
^^^^^^^^^^^^^^^

Spatial patches are used to get uncertainty estimates using jackknife or
bootstrap resampling. Furthermore, they allow to parallelise the correlation
measurements by distributing pairs of patches to different CPUs. Depending on
the backend, the number of jobs depends on the patch size compared to the
measurement scales. The larger the scales, the more neighbouring patches must
be visited, in the worst case this scales with the number of patches squared.

Construction
""""""""""""

Patches should ideally not overlap spatially, not only from a performance
perspective. If the patches are a random subsample of the full data set, the
spatial resampling will not capture all of the spatial sample variance of the
data sets and instead be limited to the shot-noise component.

One way to construct patches is to use the built-in k-means clustering
algorithm, which will split the data into roughly similar Voroni cells
(depending on the data footprint and masking). This can be achieved by supplying
the ``npatches`` parameter at catalog construction or using the corresponding
``yaw_cli init --n-patches`` / ``n_patches`` in the ``data`` section of the YAML
configuration.

Considerations
""""""""""""""

The best choice for the number of patches depends on the data samples. The
optimum performance is achieved with a small number of patches that can be
optimally distributed on the available CPUs.

.. Note::
    The number of patches should be large enough to get a robust covariance
    estimate from spatial resampling.

Furthermore, all input catalogs should use the same patch definition.

.. Warning::

    The patch centers of all catalogs must align perfectly to produce consistent
    results.

The command line tools handle this automatically by constructing patches from
the first random catalog read in and then applying these patches automatically
to subsequent data samples.

When assigning to patches using an index column from the input data, or when
using the python API, this must ensured by the user. One way to achieve this is
passing an existing data catalog for the ``npatches`` argument when constructing
new catalogs, :ref:`as shown in the example<quickapi>`.
