Key concepts
------------

The two most important concepts of yet_another_wizz are **patches** and spatial
**caching**. While the former is mostly concerned with the performance of the
correlation function measurements, the latter is import for estimating
uncertainties.

Patches
^^^^^^^

Spatial patches are used to get uncertainty estimate using jackknife or
bootstrap resampling. 

Caching
^^^^^^^

Caching is used to speed up sharing data 
Caching is used to circumvent the performance bottleneck of multiprocessing,
