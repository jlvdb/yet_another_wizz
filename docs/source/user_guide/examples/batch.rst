.. _quickbatch:

Using batch processing
----------------------

As mentioned at the end of the :ref:`previous tutoral<projoutputs>`, the
command line tools produce YAML files that allow it to easily reproduce a
project. This is the purpose of the batch processing :ref:`command<yaw_run>`
``yaw run``.

All we need to do is run this command on the YAML configuration file (see
contents below) found in the previously created project ``output``:

.. code-block:: console

    yaw run reproduced -s output/setup.yaml

The results will be written to a new project folder called ``reproduced`` and
will contain the same files in the end, including a copy of the original YAML
configuration. More details on configuration files can be found in the section
on :ref:`batch processing<yaw_run>`.

.. literalinclude:: example.yaml
    :caption: ``output/setup.yaml``
    :language: yaml
