import numpy as np
from numpy import testing as npt
from pytest import fixture, raises

from yaw import coordinates


def test_sgn():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    npt.assert_equal(coordinates.sgn(values), result)


@fixture
def data_sky_ra_sky():
    return np.deg2rad([
        [-90., 0.],
        [  0., 0.],
        [ 90., 0.],
        [180., 0.],
        [270., 0.],
        [360., 0.],
        [450., 0.]])


@fixture
def data_sky_ra_3d():
    return np.array([
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.]])


@fixture
def data_sky_dec_sky():
    return np.deg2rad([
        [-90.,  90.],
        [  0.,  90.],
        [ 90.,  90.],
        [-90., -90.],
        [  0., -90.],
        [ 90., -90.]])


@fixture
def data_sky_dec_3d():
    return  np.array([
        [0., 0.,  1.],
        [0., 0.,  1.],
        [0., 0.,  1.],
        [0., 0., -1.],
        [0., 0., -1.],
        [0., 0., -1.]])


class TestCoordSky:

    def test_values(self, data_sky_ra_sky):
        # array of coordinates
        ra, dec = data_sky_ra_sky.T
        coords = coordinates.CoordSky(ra=ra.tolist(), dec=dec.tolist())
        npt.assert_equal(ra, coords.ra)
        npt.assert_equal(dec, coords.dec)
        npt.assert_equal(data_sky_ra_sky, coords.values)
        assert coords.ndim == 2
        assert len(coords) == len(data_sky_ra_sky)
        # single coordinate
        ra, dec = data_sky_ra_sky[0]
        coords = coordinates.CoordSky(ra=ra, dec=dec)
        npt.assert_equal([ra], coords.ra)
        npt.assert_equal([dec], coords.dec)
        npt.assert_equal(data_sky_ra_sky[0], coords.values)
        assert len(coords) == 1
        coords.__repr__()

    def test_from_array(self, data_sky_ra_sky):
        # array of coordinates
        coords = coordinates.CoordSky.from_array(data_sky_ra_sky)
        npt.assert_equal(data_sky_ra_sky, coords.values)
        # single coordinate
        coords = coordinates.CoordSky.from_array(data_sky_ra_sky[0])
        npt.assert_equal(data_sky_ra_sky[0], coords.values)

    def test_from_coords_iter(self, data_sky_ra_sky):
        # array of coordinates
        coord_list = []
        for ra, dec in data_sky_ra_sky:
            coord_list.append(coordinates.CoordSky(ra, dec))
        coords = coordinates.CoordSky.from_coords(coord_list)
        npt.assert_equal(data_sky_ra_sky, coords.values)
        for coord1, coord2 in zip(coords, coord_list):
            assert coord1.ra == coord2.ra

    def test_slicing(self, data_sky_ra_sky):
        ra, dec = data_sky_ra_sky.T
        coords = coordinates.CoordSky(ra=ra, dec=dec)
        idx = slice(-1, 2, -1)
        npt.assert_equal(ra[idx], coords.ra[idx])
        npt.assert_equal(dec[idx], coords.dec[idx])

    def test_mean(self):
        mean = coordinates.CoordSky([-1, 0, 1], [-1, 0, 1]).mean()
        assert mean.ra == 0.0
        assert mean.dec == 0.0

    def test_conversion(
            self, data_sky_ra_sky, data_sky_ra_3d,
            data_sky_dec_sky, data_sky_dec_3d):
        # along constant RA
        ra, dec = data_sky_ra_sky.T
        x, y, z = data_sky_ra_3d.T
        coords = coordinates.CoordSky(ra, dec)
        result = coords.to_3d()
        npt.assert_allclose(x, result.x, atol=1e-15)
        npt.assert_allclose(y, result.y, atol=1e-15)
        npt.assert_allclose(z, result.z, atol=1e-15)
        # along constant Dec
        ra, dec = data_sky_dec_sky.T
        x, y, z = data_sky_dec_3d.T
        coords = coordinates.CoordSky(ra, dec)
        result = coords.to_3d()
        npt.assert_allclose(x, result.x, atol=1e-15)
        npt.assert_allclose(y, result.y, atol=1e-15)
        npt.assert_allclose(z, result.z, atol=1e-15)
        # test one single point
        ra, dec = data_sky_ra_sky[0].T
        x, y, z = data_sky_ra_3d[0].T
        coords = coordinates.CoordSky(ra, dec)
        result = coords.to_3d()
        npt.assert_allclose(x, result.x, atol=1e-15)
        npt.assert_allclose(y, result.y, atol=1e-15)
        npt.assert_allclose(z, result.z, atol=1e-15)

        # trivial case
        assert coords is coords.to_sky()

    def test_distance(self):
        # measure across pole
        a = np.deg2rad([ 90, 45])
        b = np.deg2rad([270, 45])
        npt.assert_allclose(
            coordinates.CoordSky(*a).distance(coordinates.CoordSky(*b)).values,
            np.pi/2)
        # test RA periodicity
        a = np.deg2rad([-90, -45])
        b = np.deg2rad([270,  45])
        npt.assert_allclose(
            coordinates.CoordSky(*a).distance(coordinates.CoordSky(*b)).values,
            np.pi/2)
        # test poles
        a = np.deg2rad([0, -90])
        b = np.deg2rad([10, 90])
        npt.assert_allclose(
            coordinates.CoordSky(*a).distance(coordinates.CoordSky(*b)).values,
            np.pi)
        # test array
        a = coordinates.CoordSky([0.0, 0.0], [0.0, np.pi/2])
        b = coordinates.CoordSky([0.0, 0.0], [0.0, 0.0])
        npt.assert_allclose(a.distance(b).values, [0.0, np.pi/2])
        a = coordinates.CoordSky([0.0, 0.0], [0.0, np.pi/2])
        b = coordinates.CoordSky(0.0, 0.0)
        npt.assert_allclose(a.distance(b).values, [0.0, np.pi/2])


@fixture
def data_3d_ra_3d():
    return np.array([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.],
        [ 0., -1.,  0.]])


@fixture
def data_3d_ra_sky():
    return np.deg2rad([
        [  0., 0.],
        [ 90., 0.],
        [180., 0.],
        [270., 0.]])


@fixture
def data_3d_dec_3d():
    return np.array([
        [0., 0.,  1.],
        [0., 0., -1.]])


@fixture
def data_3d_dec_sky():
    return np.deg2rad([
        [0.,  90.],
        [0., -90.]])


class TestCoord3D:

    def test_values(self, data_3d_ra_3d):
        # array of coordinates
        x, y, z = data_3d_ra_3d.T
        coords = coordinates.Coord3D(x=x.tolist(), y=y.tolist(), z=z.tolist())
        npt.assert_equal(x, coords.x)
        npt.assert_equal(y, coords.y)
        npt.assert_equal(z, coords.z)
        npt.assert_equal(data_3d_ra_3d, coords.values)
        assert coords.ndim == 3
        assert len(coords) == len(data_3d_ra_3d)
        # single coordinate
        x, y, z = data_3d_ra_3d[0]
        coords = coordinates.Coord3D(x=x, y=y, z=z)
        npt.assert_equal([x], coords.x)
        npt.assert_equal([y], coords.y)
        npt.assert_equal([z], coords.z)
        npt.assert_equal(data_3d_ra_3d[0], coords.values)
        assert len(coords) == 1
        coords.__repr__()

    def test_from_array(self, data_sky_ra_3d):
        # array of coordinates
        coords = coordinates.Coord3D.from_array(data_sky_ra_3d)
        npt.assert_equal(data_sky_ra_3d, coords.values)
        # single coordinate
        coords = coordinates.Coord3D.from_array(data_sky_ra_3d[0])
        npt.assert_equal(data_sky_ra_3d[0], coords.values)

    def test_from_coords_iter(self, data_sky_ra_3d):
        # array of coordinates
        coord_list = []
        for x, y, z in data_sky_ra_3d:
            coord_list.append(coordinates.Coord3D(x, y, z))
        coords = coordinates.Coord3D.from_coords(coord_list)
        npt.assert_equal(data_sky_ra_3d, coords.values)
        for coord1, coord2 in zip(coords, coord_list):
            assert coord1.x == coord2.x

    def test_slicing(self, data_3d_ra_3d):
        x, y, z = data_3d_ra_3d.T
        coords = coordinates.Coord3D(x=x, y=y, z=z)
        idx = slice(-1, 2, -1)
        coords = coords[idx]
        npt.assert_equal(x[idx], coords.x)
        npt.assert_equal(y[idx], coords.y)
        npt.assert_equal(z[idx], coords.z)

    def test_mean(self):
        mean = coordinates.Coord3D([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]).mean()
        assert mean.x == 0.0
        assert mean.y == 0.0
        assert mean.z == 0.0

    def test_conversion(
            self, data_3d_ra_3d, data_3d_ra_sky,
            data_3d_dec_3d, data_3d_dec_sky):
        # along constant RA
        x, y, z = data_3d_ra_3d.T
        ra, dec = data_3d_ra_sky.T
        coords = coordinates.Coord3D(x=x, y=y, z=z)
        result = coords.to_sky()
        npt.assert_allclose(ra, result.ra, atol=1e-15)
        npt.assert_allclose(dec, result.dec, atol=1e-15)
        # along constant Dec
        x, y, z = data_3d_dec_3d.T
        ra, dec = data_3d_dec_sky.T
        coords = coordinates.Coord3D(x=x, y=y, z=z)
        result = coords.to_sky()
        npt.assert_allclose(ra, result.ra, atol=1e-15)
        npt.assert_allclose(dec, result.dec, atol=1e-15)
        # test one single point
        x, y, z = data_3d_ra_3d[0].T
        ra, dec = data_3d_ra_sky[0].T
        coords = coordinates.Coord3D(x=x, y=y, z=z)
        result = coords.to_sky()
        npt.assert_allclose(ra, result.ra, atol=1e-15)
        npt.assert_allclose(dec, result.dec, atol=1e-15)

        # trivial case
        assert coords is coords.to_3d()

    # test_distance
    # covered in TestCoordSky


class TestDist3D:

    def test_values(self):
        # single value
        assert coordinates.Dist3D(1).values == 1.0
        assert len(coordinates.Dist3D(1)) == 1
        # array
        dists = coordinates.Dist3D([0.1, 1.0])
        npt.assert_equal(dists.values, [0.1, 1.0])
        assert len(dists) == 2
        dists.__repr__()
        f"{coordinates.Dist3D(2.0)}"

    def test_from_dists(self):
        dists = np.array([0.1, 1.0])
        dist_list = [coordinates.Dist3D(d) for d in dists]
        npt.assert_equal(dists, coordinates.Dist3D.from_dists(dist_list).values)

    def test_iter(self):
        dists = np.array([0.1, 1.0])
        dists = coordinates.Dist3D(dists)
        for a, b in zip(dists, dists.values):
            assert a == coordinates.Dist3D(b)

    def test_ordering(self):
        assert coordinates.Dist3D(1.0) == coordinates.Dist3D(1.0)
        assert coordinates.Dist3D(1.0) <= coordinates.Dist3D(1.0)
        assert coordinates.Dist3D(1.0) < coordinates.Dist3D(2.0)
        assert coordinates.Dist3D(2.0) != coordinates.Dist3D(1.0)
        assert coordinates.Dist3D(2.0) >= coordinates.Dist3D(1.0)
        assert coordinates.Dist3D(2.0) > coordinates.Dist3D(1.0)
        # test array
        assert np.all(
            coordinates.Dist3D([0.0, 1.0]) == coordinates.Dist3D([0.0, 1.0]))

    def test_add_sub(self):
        dist_90deg = coordinates.Dist3D(np.sqrt(2.0))  # don't add linearly
        # sum to 180 degrees
        assert dist_90deg + dist_90deg == coordinates.Dist3D(2.0)
        # subtract value of itself
        assert (
            coordinates.Dist3D(1.0) - coordinates.Dist3D(1.0) ==
            coordinates.Dist3D(0.0))
        # negative separation
        npt.assert_allclose(
            (coordinates.Dist3D(0.0) - coordinates.Dist3D(1.0)).values,
            coordinates.Dist3D(-1.0).values)
        # wrap around unit sphere
        assert coordinates.Dist3D(2.0) + dist_90deg == dist_90deg

    def test_conversion(self):
        assert coordinates.Dist3D(0.0).to_sky() == coordinates.DistSky(0.0)
        # test at 90 degree separation -> sqrt(2)
        npt.assert_allclose(
            coordinates.Dist3D(np.sqrt(2.0)).to_sky().values,
            coordinates.DistSky(np.pi/2).values, atol=1e-15)
        # test at 180 degree separation -> 2
        npt.assert_allclose(
            coordinates.Dist3D(2.0).to_sky().values,
            coordinates.DistSky(np.pi).values, atol=1e-15)
        # points apart > unit sphere radius
        with raises(ValueError):
            coordinates.Dist3D(2.000001).to_sky()

        # trivial case
        assert coordinates.Dist3D(0.0) == coordinates.Dist3D(0.0).to_3d()

    def test_min_max(self):
        dists = [0., -1., 1.]
        dists = coordinates.Dist3D(dists)
        assert dists.min() == coordinates.Dist3D(-1.0)
        assert dists.max() == coordinates.Dist3D(1.0)


class TestDistSky:
    # other methods covered above

    def test_values(self):
        assert coordinates.DistSky(1).values == 1.0
        dists = coordinates.DistSky([0.1, 1.0])
        npt.assert_equal(dists.values, [0.1, 1.0])
        dists.__repr__()
        f"{coordinates.DistSky(2.0)}"

    def test_conversion(self):
        # see also TestDist3D.test_to_sky
        assert coordinates.DistSky(0.0).to_3d() == coordinates.Dist3D(0.0)
        npt.assert_allclose(
            coordinates.DistSky(np.pi/2).to_3d().values,
            coordinates.Dist3D(np.sqrt(2.0)).values, atol=1e-15)
        # test at 180 degree separation -> 2
        npt.assert_allclose(
            coordinates.DistSky(np.pi).to_3d().values,
            coordinates.Dist3D(2.0).values, atol=1e-15)
        # wrap around
        npt.assert_allclose(
            coordinates.DistSky(1.5*np.pi).to_3d().values,
            coordinates.Dist3D(np.sqrt(2.0)).values, atol=1e-15)

        # trivial case
        assert coordinates.DistSky(0.0) == coordinates.DistSky(0.0).to_sky()
