import numpy as np
from numpy import testing as npt
from pytest import fixture, raises

from yaw.core.coordinates import Coord3D, CoordSky, Dist3D, DistSky, sgn


def test_sgn():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    npt.assert_equal(sgn(values), result)


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
        coords = CoordSky(ra=ra.tolist(), dec=dec.tolist())
        npt.assert_equal(ra, coords.ra)
        npt.assert_equal(dec, coords.dec)
        npt.assert_equal(data_sky_ra_sky, coords.values)
        # single coordinate
        ra, dec = data_sky_ra_sky[0]
        coords = CoordSky(ra=ra, dec=dec)
        assert ra == coords.ra
        assert dec == coords.dec
        npt.assert_equal(data_sky_ra_sky[0], coords.values)
        coords.__repr__()

    def test_slicing(self, data_sky_ra_sky):
        ra, dec = data_sky_ra_sky.T
        coords = CoordSky(ra=ra, dec=dec)
        idx = slice(-1, 2, -1)
        npt.assert_equal(ra[idx], coords.ra[idx])
        npt.assert_equal(dec[idx], coords.dec[idx])

    def test_mean(self):
        mean = CoordSky([-1, 0, 1], [-1, 0, 1]).mean()
        assert mean.ra == 0.0
        assert mean.dec == 0.0

    def test_conversion(
            self, data_sky_ra_sky, data_sky_ra_3d,
            data_sky_dec_sky, data_sky_dec_3d):
        # along constant RA
        ra, dec = data_sky_ra_sky.T
        x, y, z = data_sky_ra_3d.T
        coords = CoordSky(ra, dec)
        result = coords.to_3d()
        npt.assert_allclose(x, result.x, atol=1e-15)
        npt.assert_allclose(y, result.y, atol=1e-15)
        npt.assert_allclose(z, result.z, atol=1e-15)
        # along constant Dec
        ra, dec = data_sky_dec_sky.T
        x, y, z = data_sky_dec_3d.T
        coords = CoordSky(ra, dec)
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
            CoordSky(*a).distance(CoordSky(*b)).values,
            np.pi/2)
        # test RA periodicity
        a = np.deg2rad([-90, -45])
        b = np.deg2rad([270,  45])
        npt.assert_allclose(
            CoordSky(*a).distance(CoordSky(*b)).values,
            np.pi/2)
        # test poles
        a = np.deg2rad([0, -90])
        b = np.deg2rad([10, 90])
        npt.assert_allclose(
            CoordSky(*a).distance(CoordSky(*b)).values,
            np.pi)
        # test array
        a = CoordSky([0.0, 0.0], [0.0, np.pi/2])
        b = CoordSky([0.0, 0.0], [0.0, 0.0])
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
        coords = Coord3D(x=x.tolist(), y=y.tolist(), z=z.tolist())
        npt.assert_equal(x, coords.x)
        npt.assert_equal(y, coords.y)
        npt.assert_equal(z, coords.z)
        npt.assert_equal(data_3d_ra_3d, coords.values)
        # single coordinate
        x, y, z = data_3d_ra_3d[0]
        coords = Coord3D(x=x, y=y, z=z)
        assert x == coords.x
        assert z == coords.y
        assert z == coords.z
        npt.assert_equal(data_3d_ra_3d[0], coords.values)
        # just try to run
        coords.__repr__()

    def test_slicing(self, data_3d_ra_3d):
        x, y, z = data_3d_ra_3d.T
        coords = Coord3D(x=x, y=y, z=z)
        idx = slice(-1, 2, -1)
        coords = coords[idx]
        npt.assert_equal(x[idx], coords.x)
        npt.assert_equal(y[idx], coords.y)
        npt.assert_equal(z[idx], coords.z)

    def test_mean(self):
        mean = Coord3D([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]).mean()
        assert mean.x == 0.0
        assert mean.y == 0.0
        assert mean.z == 0.0

    def test_conversion(
            self, data_3d_ra_3d, data_3d_ra_sky,
            data_3d_dec_3d, data_3d_dec_sky):
        # along constant RA
        x, y, z = data_3d_ra_3d.T
        ra, dec = data_3d_ra_sky.T
        coords = Coord3D(x=x, y=y, z=z)
        result = coords.to_sky()
        npt.assert_allclose(ra, result.ra, atol=1e-15)
        npt.assert_allclose(dec, result.dec, atol=1e-15)
        # along constant Dec
        x, y, z = data_3d_dec_3d.T
        ra, dec = data_3d_dec_sky.T
        coords = Coord3D(x=x, y=y, z=z)
        result = coords.to_sky()
        npt.assert_allclose(ra, result.ra, atol=1e-15)
        npt.assert_allclose(dec, result.dec, atol=1e-15)

        # trivial case
        assert coords is coords.to_3d()

    # test_distance
    # covered in TestCoordSky


class TestDist3D:

    def test_values(self):
        assert Dist3D(1).values == 1.0
        dists = Dist3D([0.1, 1.0])
        npt.assert_equal(dists.values, [0.1, 1.0])
        dists.__repr__()
        f"{Dist3D(2.0)}"

    def test_ordering(self):
        assert Dist3D(1.0) == Dist3D(1.0)
        assert Dist3D(1.0) <= Dist3D(1.0)
        assert Dist3D(1.0) < Dist3D(2.0)
        assert Dist3D(2.0) != Dist3D(1.0)
        assert Dist3D(2.0) >= Dist3D(1.0)
        assert Dist3D(2.0) > Dist3D(1.0)
        # test array
        assert np.all(Dist3D([0.0, 1.0]) == Dist3D([0.0, 1.0]))

    def test_add_sub(self):
        dist_90deg = Dist3D(np.sqrt(2.0))  # don't add linearly
        # sum to 180 degrees
        assert dist_90deg + dist_90deg == Dist3D(2.0)
        # subtract value of itself
        assert Dist3D(1.0) - Dist3D(1.0) == Dist3D(0.0)
        # negative separation
        npt.assert_allclose(
            (Dist3D(0.0) - Dist3D(1.0)).values,
            Dist3D(-1.0).values)
        # wrap around unit sphere
        assert Dist3D(2.0) + dist_90deg == dist_90deg

    def test_to_sky(self):
        assert Dist3D(0.0).to_sky() == DistSky(0.0)
        # test at 90 degree separation -> sqrt(2)
        npt.assert_allclose(
            Dist3D(np.sqrt(2.0)).to_sky().values,
            DistSky(np.pi/2).values, atol=1e-15)
        # test at 180 degree separation -> 2
        npt.assert_allclose(
            Dist3D(2.0).to_sky().values,
            DistSky(np.pi).values, atol=1e-15)
        # points apart > unit sphere radius
        with raises(ValueError):
            Dist3D(2.000001).to_sky()


class TestDistSky:
    # other methods covered above

    def test_values(self):
        assert DistSky(1).values == 1.0
        dists = DistSky([0.1, 1.0])
        npt.assert_equal(dists.values, [0.1, 1.0])
        dists.__repr__()
        f"{DistSky(2.0)}"

    def test_to_3d(self):
        # see also TestDist3D.test_to_sky
        assert DistSky(0.0).to_3d() == Dist3D(0.0)
        npt.assert_allclose(
            DistSky(np.pi/2).to_3d().values,
            Dist3D(np.sqrt(2.0)).values, atol=1e-15)
        # test at 180 degree separation -> 2
        npt.assert_allclose(
            DistSky(np.pi).to_3d().values,
            Dist3D(2.0).values, atol=1e-15)
        # wrap around
        npt.assert_allclose(
            DistSky(1.5*np.pi).to_3d().values,
            Dist3D(np.sqrt(2.0)).values, atol=1e-15)
