import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import fixture, raises

from yaw.coordinates import AngularCoordinates, AngularDistances, sgn


def test_sgn():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    assert_array_equal(sgn(values), result)


@fixture(name="coords_ra_dec_fixed_dec")
def fixture_coords_ra_dec_fixed_dec():
    return np.deg2rad(
        [  # RA    Dec
            [-90.0, 0.0],
            [0.0, 0.0],
            [90.0, 0.0],
            [180.0, 0.0],
            [270.0, 0.0],
            [360.0, 0.0],
            [450.0, 0.0],
        ]
    )


@fixture(name="coords_xyz_fixed_dec")
def fixture_coords_xyz_fixed_dec():
    return np.array(
        [  # x     y    z
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


@fixture(name="coords_ra_dec_fixed_ra")
def fixture_coords_ra_dec_fixed_ra():
    return np.deg2rad(
        [
            [-90.0, 90.0],
            [0.0, 90.0],
            [90.0, 90.0],
            [-90.0, -90.0],
            [0.0, -90.0],
            [90.0, -90.0],
        ]
    )


@fixture(name="coords_xyz_fixed_ra")
def fixture_coords_xyz_fixed_ra():
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )


class TestAngularCoordinates:
    def test_init(self, coords_ra_dec_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert_array_equal(coords.data, coords_ra_dec_fixed_dec)

        repr(coords)

    def test_init_casting(self, coords_ra_dec_fixed_dec):
        data = coords_ra_dec_fixed_dec.astype(np.float32)
        coords = AngularCoordinates(data)
        assert_array_almost_equal(coords.data, coords_ra_dec_fixed_dec)

    def test_init_validation(self, coords_ra_dec_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec[0])
        assert coords.data.shape == (1, 2)

        with raises(ValueError, match=".*dimensions.*"):
            AngularCoordinates(np.ones(3))

        assert coords.data.shape == (1, 2)

    def test_prop_ra(self, coords_ra_dec_fixed_dec):
        expect = coords_ra_dec_fixed_dec[:, 0]
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert_array_equal(expect, coords.ra)

    def test_prop_dec(self, coords_ra_dec_fixed_ra):
        expect = coords_ra_dec_fixed_ra[:, 1]
        coords = AngularCoordinates(coords_ra_dec_fixed_ra)
        assert_array_equal(expect, coords.dec)

    def test_eq(self, coords_ra_dec_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert np.all(coords == coords)
        assert coords != 1

    def test_from_coords(self, coords_ra_dec_fixed_ra):
        expect = AngularCoordinates(coords_ra_dec_fixed_ra)
        coords = AngularCoordinates.from_coords(iter(expect))
        assert np.all(expect == coords)

    def test_from_3d_fixed_ra(self, coords_ra_dec_fixed_ra, coords_xyz_fixed_ra):
        # NOTE: RAs will not be recovered at coordinate singularity
        decs = coords_ra_dec_fixed_ra[:, 1]
        expect = np.column_stack([np.zeros_like(decs), decs])
        coords = AngularCoordinates.from_3d(coords_xyz_fixed_ra)
        assert_array_almost_equal(expect, coords.data)

    def test_from_3d_fixed_dec(self, coords_ra_dec_fixed_dec, coords_xyz_fixed_dec):
        # NOTE: RAs will be wrapped into [0, 2pi)
        ras = coords_ra_dec_fixed_dec[:, 0] % (2.0 * np.pi)
        expect = np.column_stack([ras, np.zeros_like(ras)])
        coords = AngularCoordinates.from_3d(coords_xyz_fixed_dec)
        assert_array_almost_equal(expect, coords.data)

    def test_to_3d_fixed_ra(self, coords_ra_dec_fixed_ra, coords_xyz_fixed_ra):
        coords = AngularCoordinates(coords_ra_dec_fixed_ra)
        assert_array_almost_equal(coords.to_3d(), coords_xyz_fixed_ra)

    def test_to_3d_fixed_dec(self, coords_ra_dec_fixed_dec, coords_xyz_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert_array_almost_equal(coords.to_3d(), coords_xyz_fixed_dec)

    def test_mean(self, coords_ra_dec_fixed_dec):
        expect = AngularCoordinates([0.0, 0.0])
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert_array_equal(expect, coords.mean())

    def test_distance(self, coords_ra_dec_fixed_dec):
        ref_point = AngularCoordinates([0.0, 0.0])
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        expect = np.array([0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5]) * np.pi

        dists = ref_point.distance(coords)
        assert_array_almost_equal(dists.data, expect)

        dists = coords.distance(ref_point)
        assert_array_almost_equal(dists.data, expect)

        with raises(TypeError):
            ref_point.distance(1.0)

    def test_copy(self, coords_ra_dec_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert id(coords) is not id(coords.copy())
        assert np.all(coords == coords.copy())

    def test_to_list(self, coords_ra_dec_fixed_dec):
        coords = AngularCoordinates(coords_ra_dec_fixed_dec)
        assert coords.tolist() == coords.data.tolist()


@fixture(name="chord_dists")
def fixture_chord_dists():
    return np.array([0.0, np.sqrt(2.0), 2.0])


@fixture(name="angular_dists")
def fixture_angular_dists():
    return np.array([0.0, 0.5, 1.0]) * np.pi


class TestAngularDistances:
    def test_init(self):
        expect = np.array([1.0], dtype=np.float64)
        dists = AngularDistances(1.0)
        assert_array_equal(expect, dists.data)

    def test_init_casting(self, coords_ra_dec_fixed_dec):
        expect = np.array([1.0], dtype=np.float64)
        data = expect[0].astype(np.float32)
        dists = AngularDistances(data)
        assert_array_almost_equal(expect, dists.data)

    def test_from_dists(self):
        expect = AngularDistances(np.linspace(0.0, 2.0, 10))
        dists = AngularDistances.from_dists(iter(expect))
        assert np.all(expect == dists)

    def test_from_3d(self, chord_dists, angular_dists):
        dists = AngularDistances.from_3d(chord_dists)
        assert_array_almost_equal(angular_dists, dists.data)

        with raises(ValueError, match=".*exceeds.*"):
            AngularDistances.from_3d(2.0 + 1e-12)

    def test_to_3d(self, chord_dists, angular_dists):
        dists = AngularDistances(angular_dists)
        assert_array_almost_equal(chord_dists, dists.to_3d())

    def test_ordering(self):
        assert AngularDistances(1.0) == AngularDistances(1.0)
        assert AngularDistances(1.0) != AngularDistances(2.0)

        assert AngularDistances(1.0) < AngularDistances(2.0)
        assert AngularDistances(1.0) <= AngularDistances(1.0)
        with raises(TypeError):
            AngularDistances(1.0) < 2.0

        assert AngularDistances(2.0) > AngularDistances(1.0)
        assert AngularDistances(2.0) >= AngularDistances(2.0)

    def test_add(self):
        assert AngularDistances(1.0) + AngularDistances(1.0) == AngularDistances(2.0)

        with raises(TypeError):
            AngularDistances(1.0) + 1.0

    def test_sub(self):
        assert AngularDistances(2.0) - AngularDistances(1.0) == AngularDistances(1.0)

        with raises(TypeError):
            AngularDistances(2.0) - 1.0

    def test_min(self):
        data = np.linspace(0.0, 1.0)
        dists = AngularDistances(data)
        assert dists.min() == data.min()

    def test_max(self):
        data = np.linspace(0.0, 1.0)
        dists = AngularDistances(data)
        assert dists.max() == data.max()
