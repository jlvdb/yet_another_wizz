Correlation Functions
=====================

.. currentmodule:: yaw


The functions that measure correlations return a special container that stores
pair counts per redshift bin and spatial patches:

.. autosummary::
    :toctree: autogen

    CorrFunc


The pair counts stored in the class above are wrapped into another set of
containers that store the normalised pair counts per patch and redshift, which
in turn store the raw pair counts and the size of the patches (used to compute
the normalisation factor for the pair counts):

.. autosummary::
    :toctree: autogen

    paircounts.NormalisedCounts
    paircounts.PatchedCounts
    paircounts.PatchedTotals
