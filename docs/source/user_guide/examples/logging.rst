Logging
-------

Most operations that `yet_another_wizz` performs are logged. To access these
logs, e.g. on terminal or by redirecting them to a file, use the following code:

.. code-block:: python

    from yaw.utils import get_logger

    get_logger(
        "info"  # default log level
        stdout=True,  # by default, write messages to stdout
        # file=None  # additionally write log messages with time stamp to a file
    )


.. tab-set::

   .. tab-item:: Terminal logging

      .. image:: /_static/logs_pretty.png

   .. tab-item:: Text file logging

      .. image:: /_static/logs_plain.png
