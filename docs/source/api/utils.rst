Utilities
---------

.. currentmodule:: yaw

Most `yet_another_wizz` code produces detailed log messages. These can be
filtered and collected easily on standard output or in a log file using the
following utility function:

.. autosummary::
    :toctree: autogen

    utils.get_logger


There are two utility classes in `yet_another_wizz` that are used to handle
angular coordinates and distance computations in angular and 3-dimensional
Euclidean coordintes:

.. autosummary::
    :toctree: autogen

    AngularCoordinates
    AngularDistances


Additionally, there is a special container with convenience methods that store
bin intervals:

.. autosummary::
    :toctree: autogen

    Binning
