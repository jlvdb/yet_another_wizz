.. _quickbatch:

Using batch processing
----------------------

As mentioned at the end of the :ref:`previous tutoral<projoutputs>`, the
command line tools produce YAML files that allow it to easily reproduce a
project. This is the purpose of the batch processing :ref:`command<yaw_run>`
``yaw_cli run``.

All we need to do is run this command on the YAML configuration file (see
contents below) found in the previously created project ``output``:

.. code-block:: bash

    $ yaw_cli run reproduced -s output/setup.yaml

The results will be written to a new project folder called ``reproduced`` and
will contain the same files in the end, including a copy of the original YAML
configuration. More details on configuration files can be found in the section
on :ref:`batch processing<yaw_run>`.

.. dropdown:: Contents of ``output/setup.yaml``.

    .. literalinclude:: example.yaml
        :language: yaml


Processing subsets
^^^^^^^^^^^^^^^^^^

As in the previous example, we can also process the unknown sample in subsets if
these subsets are stored in separate input catalogues. In that case, the syntax
of the configuration file from above changes slightly since we must assign an
index to each of the unknown data input catalogues. These bin indices can be
assinged arbitrarily, but to be consistent with the ``yaw_cli cross`` example
the configuration file would have to look like this:

.. dropdown:: Contents of ``output/setup.yaml`` with tomographic bins.

    .. literalinclude:: example_bins.yaml
        :language: yaml

.. Note::

    To provide a multiple unknown data subsets the single file path must be
    replaced by a mapping of bin index to file path, the rest of the file is
    unchanged compared to the example above.
