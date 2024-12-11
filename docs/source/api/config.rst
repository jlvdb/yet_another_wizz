Configuration
-------------

.. currentmodule:: yaw

There is one central class that manages all configuration parameters for the
correlation measurements. These include the correlation scales, pair weighting,
and redshift binning for the reference catalog:

.. autosummary::
    :toctree: autogen

    Configuration


The configuration above is organised hierarchically. The main configuration
separates the scale and binning configuration into two sub-classes, however
usually it is not necessary to interact with these sub-classes directly.

.. autosummary::
    :toctree: autogen

    config.ScalesConfig
    config.BinningConfig
