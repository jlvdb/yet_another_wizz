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
