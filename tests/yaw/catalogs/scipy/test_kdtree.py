from itertools import product

import numpy as np
from numpy import testing as npt
from pytest import fixture, raises

from yaw.catalogs.scipy import kdtree
from yaw.core import coordinates


@fixture
def sphere_edges():
    return np.deg2rad(
        [[0.0, 0.0], [90.0, 0.0], [180.0, 0.0], [270.0, 0.0], [0.0, 90.0], [0.0, -90.0]]
    )


@fixture
def sphere_points_rad(sphere_edges):
    def const_segment(x):
        return np.full_like(base_segment, x)

    # draw the major great circles of the x, y, and z plain
    base_segment = np.linspace(0.0, 90.0, 91)[1:-1]  # (0, 90)
    # start with the trivial points
    ra, dec = sphere_edges.T
    ras = [np.rad2deg(ra)]
    decs = [np.rad2deg(dec)]
    # z-plain great circle
    for offset in (0.0, 90.0, 180.0, 270.0):
        ras.append(base_segment + offset)
        decs.append(const_segment(0.0))
    # y-plane great circle
    for sign, ra in product([-1.0, 1.0], [0.0, 180.0]):
        ras.append(const_segment(ra))
        decs.append(sign * base_segment)
    # x-plane great circle
    for sign, ra in product([-1.0, 1.0], [90.0, 270.0]):
        ras.append(const_segment(ra))
        decs.append(sign * base_segment)
    # return in radian
    points = np.transpose([np.concatenate(ras), np.concatenate(decs)])
    return np.deg2rad(points)


@fixture
def sphere_points_xyz(sphere_points_rad):
    ra, dec = sphere_points_rad.T
    return coordinates.CoordSky(ra, dec).to_3d().values


@fixture
def scales_window():
    scales = []
    for lower in np.arange(179):
        scales.append(np.deg2rad([lower, lower + 1.0]))
    return np.array(scales) + 0.0001  # avoid numerical precision issues


@fixture
def scales_culmulative():
    scales = []
    for upper in np.arange(1, 180):
        scales.append(np.deg2rad([0.0, upper]))
    return np.array(scales) + 0.0001  # avoid numerical precision issues


@fixture
def counts_window():
    return np.array(
        [
            360.0 if lower == 89.0 else 4.0  # 360 points at equator, else 4
            for lower in np.arange(179)
        ]
    )


@fixture
def counts_culmulative(counts_window):
    return np.cumsum(counts_window)


@fixture
def tree(sphere_points_rad):
    positions = coordinates.CoordSky(*sphere_points_rad.T)
    return kdtree.SphericalKDTree(positions)


class TestSphericalKDTree:
    def test_init(self):
        weight = np.array([1.0])
        # test with sky coordinates
        positions = coordinates.CoordSky([0.0], [0.0])
        tree = kdtree.SphericalKDTree(positions, weight)
        npt.assert_equal(tree.tree.data, np.array([[1.0, 0.0, 0.0]]))
        # test with unit sphere coordinates
        positions = coordinates.Coord3D([1.0], [0.0], [0.0])
        tree = kdtree.SphericalKDTree(positions, weight)
        npt.assert_equal(tree.tree.data, np.array([[1.0, 0.0, 0.0]]))
        # test weights and length
        assert tree.weights == weight
        assert len(tree) == len(weight)

    def test_empty_tree(self, tree):
        null_coord = coordinates.CoordSky([], [])
        null_tree = kdtree.SphericalKDTree(null_coord)
        tree.count(null_tree, [0.1, 1.0])

    def test_total(self, sphere_points_rad):
        positions = coordinates.CoordSky(*sphere_points_rad.T)
        tree = kdtree.SphericalKDTree(positions)
        assert tree.total == len(sphere_points_rad)
        weights = np.random.uniform(0, 1, size=len(sphere_points_rad))
        tree = kdtree.SphericalKDTree(positions, weights)
        assert tree.total == weights.sum()

    def test_scales_validation(self, tree):
        with raises(kdtree.InvalidScalesError, match=r".*positive.*"):
            tree.count(tree, [0.0, 1.0])
        with raises(kdtree.InvalidScalesError, match=r".*exceed.*"):
            tree.count(tree, [1.0, 4.0])
        with raises(kdtree.InvalidScalesError, match=r".*length 2.*"):
            tree.count(tree, 1.0)
        with raises(kdtree.InvalidScalesError, match=r".*length 2.*"):
            tree.count(tree, [1.0])
        with raises(kdtree.InvalidScalesError, match=r".*length 2.*"):
            tree.count(tree, [1.0, 2.0, 3.0])

    def test_count_single_window(
        self, sphere_edges, tree, scales_window, counts_window
    ):
        for ra, dec in sphere_edges:
            point = kdtree.SphericalKDTree(coordinates.CoordSky(ra, dec))
            for scales, counts in zip(scales_window, counts_window):
                assert tree.count(point, scales) == np.array([counts])

    def test_count_multi_window(self, sphere_edges, tree, scales_window, counts_window):
        for ra, dec in sphere_edges:
            point = kdtree.SphericalKDTree(coordinates.CoordSky(ra, dec))
            npt.assert_equal(tree.count(point, scales_window), counts_window)

    def test_count_single_cumulative(
        self, sphere_edges, tree, scales_culmulative, counts_culmulative
    ):
        for ra, dec in sphere_edges:
            point = kdtree.SphericalKDTree(coordinates.CoordSky(ra, dec))
            for scales, counts in zip(scales_culmulative, counts_culmulative):
                assert tree.count(point, scales) == np.array([counts])

    def test_count_multi_cumulative(
        self, sphere_edges, tree, scales_culmulative, counts_culmulative
    ):
        for ra, dec in sphere_edges:
            point = kdtree.SphericalKDTree(coordinates.CoordSky(ra, dec))
            npt.assert_equal(tree.count(point, scales_culmulative), counts_culmulative)

    def test_count_weighted(self, tree, sphere_points_rad):
        positions = coordinates.CoordSky(*sphere_points_rad.T)
        factor = 3.0  # counts should multiply by factor**2
        tree_w = kdtree.SphericalKDTree(
            positions, weights=np.full(len(sphere_points_rad), factor)
        )
        # compute the reference counts
        scale = [0.1, 1.0]
        counts = tree.count(tree, scale)
        npt.assert_equal(tree.count(tree_w, scale), counts * factor)
        npt.assert_equal(tree_w.count(tree, scale), counts * factor)
        npt.assert_equal(tree_w.count(tree_w, scale), counts * factor * factor)

    def test_count_scale_weight(self, tree, scales_culmulative):
        """currently not checking the internal weight computation"""
        no_weight = tree.count(tree, scales_culmulative)
        unit_weight = tree.count(tree, scales_culmulative, dist_weight_scale=0.0)
        npt.assert_equal(no_weight, unit_weight)
        # at least check that the weights don't change
        counts = tree.count(tree, [0.1, 1.0], dist_weight_scale=-1.0)
        count = int(counts[0])
        assert count == 516016  # 516016.855154
