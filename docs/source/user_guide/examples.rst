Usage example
=============

We start by providing a *minimal working example* for a very basic clustering
redshift analysis with additional correction of the reference sample bias.
There are three versions of this example which perform exactly the same tasks
but make use of the different operation modes of this code:

.. toctree::
    :maxdepth: 1

    examples/cmd
    examples/batch
    examples/api

This example assumes that the input catalogs are provided as FITS data files,
one for the reference sample (``reference.fits``) and a matching random catalog
(``randoms.fits``), and one for the unknown sample (``unknown.fits``). These
files contain right ascension and declination in degrees, named ``ra`` and
``dec``, as well as optional redshifts ``z``.

.. Note::

    Unfortunately we currently cannot ship the data files used in this example
    which means that there is no reference result that the user can reproduce.
