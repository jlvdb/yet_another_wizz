Coordinates
===========

*yet_another_wizz* uses two representations of coordinates internally:

- angular coordinates :math:`(\alpha, \delta)` - right ascension, declination -
  **in radian**, and
- dimensionless Cartesian coordinates :math:`(x, y, z)`

These are implemented in the coordinate classes
:class:`~yaw.coordinates.CoordSky` and :class:`~yaw.coordinates.Coord3D`

.. code-block:: python

    >>> from yaw import coordinates


Coordinate objects
------------------

Coordinates can be constructed from scalars and arrays, but are represented as
arrays internally in both cases:

.. code-block:: python

    >>> from numpy import pi

    >>> p1 = coordinates.CoordSky(ra=0.0, dec=0.0)
    >>> p1
    CoordSky(ra=array([0.]), dec=array([0.])

    >>> p2 = coordinates.CoordSky(ra=[pi/2, pi], dec=[0.0, 0])
    >>> p2
    CoordSky(ra=array([1.57079633, 3.14159265]), dec=array([0., 0.]))

Coordiantes can be easily converted from one coordante system to the other,
regardless of the coordinates system they are in:

.. code-block:: python

    >>> p1.to_3d()
    Coord3D(x=array([1.]), y=array([0.]), z=array([0.]))
    >>> p1.to_sky()
    CoordSky(ra=array([0.]), dec=array([0.])

.. Caution::

    The magnitude of Cartesian coordinates is not conserved when transforming
    to angular coordinates and back:

    .. code-block:: python

        >>> p3 = coordinates.Coord3D(10, 0, 0)
        >>> p3.to_sky().to_3d().x == p3.x
        False


The coordinate components ``ra``/``dec`` or ``x``/``y``/``z`` can be accessed
directly, e.g.:

    >>> p2.ra
    array([1.57079633, 3.14159265])

Coordinate objects also support indexing, just as their underlying data array
do, however the result is always broadcasted to an array:

    >>> p2[0]
    CoordSky(ra=array([1.57079633]), dec=array([0.]))


Distance objects
----------------

Distances between coordinates can be easily computed. They are represented as
angular distance by the :class:`~yaw.coordinates.DistSky` class or as Cartesian
distance by the :class:`~yaw.coordinates.Dist3D` class and store the distance
value in the ``value`` attribute.

.. code-block:: python

    >>> p2.distance(p1)
    DistSky([1.57079633 3.14159265])

    >>> d = p2.to_3d().distance(p1)
    >>> d
    Dist3D([1.41421356 2.        ])

    >>> d.value
    array([1.41421356 2.        ])

.. Note::

    The type of the coordiante determines the type of the distance returned,
    regardless of the coordinate system of argument of
    :meth:`~yaw.coordinates.Coordinate.distance`.

    Distances of coordiantes can only be computed if at least one coordinate
    contains a single point or both objects contain the same number of
    corrdinates.

Distance objects follow the same coordante transformation rules as coordinates:

.. code-block:: python

    >>> d.to_sky()
    DistSky([1.57079633 3.14159265])

Furthermore, they also support indexing and iteration, and additionally
addition, and subtraction, e.g.:

.. code-block:: python

    >>> DistSky(pi/2) + DistSky(pi/2)
    DistSky(3.141592653589793)

    >>> DistSky(pi/2) - DistSky(pi)
    DistSky(-1.5707963267948966)

.. Caution::

    Conversion between distances fails or is modulo :math:`2 \pi`:

    .. code-block:: python

        >>> coordinates.Dist3D(10).to_sky()
        ValueError: distance exceeds size of unit sphere

        >>> coordinates.DistSky(2*pi).to_3d()
        Dist3D(2.4492935982947064e-16)
