Python API tutorial
===================

First import the python package

>>> import yaw

Create a new configuration object for the correlation measurements

>>> config = yaw.Configuration.create(
...     rmin=100, rmax=1000, zmin=0.07, zmax=1.42)

Create a new catalog factory

>>> factory = yaw.NewCatalog(backend="scipy")

Load the reference data sample and a matching random catalog

>>> reference = factory.from_file(
...     "reference.fits", ra="ra", dec="dec", redshift="z", patches=n_patches)

Load the unknown data sample

>>> unknown = factory.from_file(
...     "unknown.fits", ra="ra", dec="dec", patches=reference)
>>> randoms = factory.from_file(
...     "ref_rand.fits", ra="ra", dec="dec", patches=reference)

Measure the cross-correlation redshfit estimate

>>> w_ss = yaw.autocorrelate(config, reference, ref_rand)
>>> w_sp = yaw.crosscorrelate(config, reference, unknown, unk_rand=randoms)
>>> n_cc = yaw.RedshiftData.from_correlation_functions(w_sp, w_ss)

