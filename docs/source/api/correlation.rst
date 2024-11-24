Correlation measurements
------------------------

.. currentmodule:: yaw

The measurements of the cross- and autocorrelation amplitudes are implemented in
two functions, which both take a configuration object, and multiple
:ref:`data catalogs<catalogs>` as input:

.. autosummary::
    :toctree: autogen

    crosscorrelate
    autocorrelate


.. _corrfunc:

Correlation functions
~~~~~~~~~~~~~~~~~~~~~

The functions above return a special container that collects and stores all
pairs counted per redshift bin and combination of spatial patches (used to
estimate covariances):

.. autosummary::
    :toctree: autogen

    CorrFunc


The container itself is a wrapper around another set of containers that store
the normalised pair counts per patch and redshift, which in turn store the raw
pair counts and metadata required to normalise the pair counts correctly:

.. autosummary::
    :toctree: autogen

    correlation.NormalisedCounts
    correlation.PatchedCounts
    correlation.PatchedSumWeights
