from itertools import product

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest import fixture, mark, raises

from yaw.binning import Binning
from yaw.catalog import trees
from yaw.catalog.patch import Patch, PatchWriter
from yaw.catalog.readers import DataChunk
from yaw.coordinates import AngularCoordinates


@mark.parametrize(
    "ang_min,ang_max",
    [
        (0.0, np.pi),
        ([0.0, 3.0], [0.1, np.pi]),
        ([0.0, 1.0], [1.0, np.pi]),
    ],
)
def test_parse_ang_limits(ang_min, ang_max):
    assert_array_equal(
        trees.parse_ang_limits(ang_min, ang_max),
        np.column_stack((ang_min, ang_max)),
    )


def test_parse_ang_limits_extra_dim():
    ang_min = [[1.0]]
    ang_max = [[10.0]]
    with raises(ValueError, match=".*1-dim.*"):
        trees.parse_ang_limits(ang_min, ang_max)


def test_parse_ang_limits_length_mismatch():
    ang_min = [1.0]
    ang_max = [10.0, 100.0]
    with raises(ValueError, match=".*length.*"):
        trees.parse_ang_limits(ang_min, ang_max)


def test_parse_ang_limits_unordered():
    ang_min = [0.0, 0.1, 0.2]
    ang_max = [0.2, 0.2, 0.2]
    with raises(ValueError, match=".*<.*"):
        trees.parse_ang_limits(ang_min, ang_max)


@mark.parametrize("ang_range", [(-0.01, 1.0), (0.0, np.pi + 1e9)])
def test_parse_ang_limits_out_of_range(ang_range):
    with raises(ValueError, match=".*not in range.*"):
        trees.parse_ang_limits(*ang_range)


@mark.parametrize(
    "ang_range,expect",
    [
        (np.array([[1.0, 10.0], [10.0, 100.0]]), [1.0, 10.0, 100.0]),
        (np.array([[1.0, 10.0], [11.0, 100.0]]), [1.0, 10.0, 11.0, 100.0]),
        (np.array([[1.0, 11.0], [10.0, 100.0]]), [1.0, 10.0, 11.0, 100.0]),
    ],
)
def test_get_ang_bins(ang_range, expect):
    result = trees.get_ang_bins(ang_range, None, None)
    assert_almost_equal(result, expect)


@mark.parametrize(
    "ang_range,expect",
    [
        (np.array([[0.1, 9.0], [9.0, 1000.0]]), [0.1, 1.0, 9.0, 10.0, 100.0, 1000.0]),
        (np.array([[0.1, 10.0], [10.0, 1000.0]]), [0.1, 1.0, 10.0, 100.0, 1000.0]),
        (
            np.array([[0.1, 10.0], [11.0, 1000.0]]),
            [0.1, 1.0, 10.0, 11.0, 100.0, 1000.0],
        ),
        (
            np.array([[0.1, 11.0], [10.0, 1000.0]]),
            [0.1, 1.0, 10.0, 11.0, 100.0, 1000.0],
        ),
    ],
)
def test_get_ang_bins_weights(ang_range, expect):
    result = trees.get_ang_bins(ang_range, 1.0, 4)
    assert_almost_equal(result, expect)


def test_logarithmic_mid():
    assert trees.logarithmic_mid([1.0, 100.0]) == 10.0

    result = trees.logarithmic_mid([1.0, 100.0, 10000.0])
    assert_array_equal(result, np.array([10.0, 1000.0]))


def test_dispatch_counts():
    data = np.arange(10)
    assert_array_equal(trees.dispatch_counts(data, cumulative=False), data[1:])


def test_dispatch_counts_cumulative():
    data = np.arange(10)
    expect = np.ones(len(data) - 1)
    assert_array_equal(trees.dispatch_counts(data, cumulative=True), expect)


@mark.parametrize(
    "ang_limits,expect",
    [
        (
            [[1.0, 1000.0]],
            [
                3,
            ],
        ),
        (
            [[10.0, 1000.0]],
            [
                2,
            ],
        ),
        ([[1.0, 10.0], [100.0, 1000.0]], [1, 1]),
        ([[1.0, 10.0], [10.0, 1000.0]], [1, 2]),
    ],
)
def test_get_counts_for_limits(ang_limits, expect):
    ang_bins = np.array([1.0, 10.0, 100.0, 1000.0])
    counts = np.ones(len(ang_bins) - 1)

    result = trees.get_counts_for_limits(counts, ang_bins, ang_limits)
    assert_array_equal(result, expect)


@fixture(name="test_points")
def fixture_test_points():
    points = np.array(
        [[0.0, 0.0], [90.0, 0.0], [180.0, 0.0], [270.0, 0.0], [0.0, 90.0], [0.0, -90.0]]
    )
    base_segment = np.arange(1.0, 90.0, 1.0)

    # z-plain great circle
    for offset in (0.0, 90.0, 180.0, 270.0):
        segment = np.column_stack(
            [base_segment + offset, np.full_like(base_segment, 0.0)]
        )
        points = np.concatenate([points, segment])

    # y-plane great circle
    for sign, ra in product([-1.0, 1.0], [0.0, 180.0]):
        segment = np.column_stack([np.full_like(base_segment, ra), sign * base_segment])
        points = np.concatenate([points, segment])

    # x-plane great circle
    for sign, ra in product([-1.0, 1.0], [90.0, 270.0]):
        segment = np.column_stack([np.full_like(base_segment, ra), sign * base_segment])
        points = np.concatenate([points, segment])

    points = np.deg2rad(points)
    return AngularCoordinates(points)


DELTA = 1e-9


class TestAngularTree:
    def test_init(self, test_points):
        tree = trees.AngularTree(test_points)
        assert tree.weights is None
        assert tree.sum_weights == float(tree.num_records)

        weights = np.random.uniform(size=len(test_points))
        tree = trees.AngularTree(test_points, weights)
        assert_array_equal(tree.weights, weights)
        assert tree.sum_weights == weights.sum()

        with raises(ValueError, match=".*shape.*"):
            trees.AngularTree(test_points, weights[1:])

        repr(tree)

    @mark.parametrize("ang_max", [1.0, 2.0, 10.0, 89.0])
    def test_count_single(self, test_points, ang_max):
        base_weight = 2.0
        weights = np.full(len(test_points), base_weight)
        tree = trees.AngularTree(test_points, weights)

        tree_single = trees.AngularTree(AngularCoordinates([0.0, 0.0]), [base_weight])

        ang_max += DELTA
        ang_min = np.deg2rad(ang_max - 1.0)
        ang_max = np.deg2rad(ang_max)
        count = tree.count(tree_single, ang_min, ang_max)

        n_points_covered = 1 * 4
        assert count == n_points_covered * base_weight**2

    @mark.parametrize("ang_max", [2.0, 10.0, 89.0])
    def test_count_bins(self, test_points, ang_max):
        base_weight = 2.0
        weights = np.full(len(test_points), base_weight)
        tree = trees.AngularTree(test_points, weights)

        tree_single = trees.AngularTree(AngularCoordinates([0.0, 0.0]), [base_weight])

        ang_max = np.arange(1.0, ang_max) + DELTA
        ang_min = np.deg2rad(ang_max - 1.0)
        ang_max = np.deg2rad(ang_max)
        count = tree.count(tree_single, ang_min, ang_max)

        n_points_covered = 1 * 4
        expect = np.full_like(ang_max, n_points_covered * base_weight**2)
        assert_array_equal(count, expect)

    @mark.parametrize("ang_max", [1.0, 2.0, 10.0, 89.0])
    def test_count_range(self, test_points, ang_max):
        base_weight = 2.0
        weights = np.full(len(test_points), base_weight)
        tree = trees.AngularTree(test_points, weights)

        tree_single = trees.AngularTree(AngularCoordinates([0.0, 0.0]), [base_weight])

        count = tree.count(tree_single, DELTA, np.deg2rad(ang_max) + DELTA)

        n_points_covered = int(ang_max) * 4
        assert count == n_points_covered * base_weight**2

    @mark.parametrize("num_bins", [1, 2])
    def test_count_empty(self, num_bins):
        no_points = np.empty((0, 2))
        tree = trees.AngularTree(AngularCoordinates(no_points), weights=[])

        ang_min = np.linspace(0.0, 1.0, num_bins) + DELTA
        ang_max = ang_min + 1.0
        count = tree.count(tree, ang_min, ang_max)

        expect = np.zeros(num_bins)
        assert_array_equal(count, expect)

    def test_count_dualtree(self, test_points):
        tree = trees.AngularTree(test_points)
        ang_lims = np.deg2rad([0.0, 1.0]) + DELTA
        count = tree.count(tree, *ang_lims)

        pole_points = 6
        segment_points = len(test_points) - pole_points
        expect = 4 * pole_points + 2 * segment_points
        assert count == expect

    def test_count_invalid_ang(self, test_points):
        tree = trees.AngularTree(test_points)
        with raises(ValueError):
            tree.count(tree, [-1.0], [1.0])
        with raises(ValueError):
            tree.count(tree, [1.0], [np.pi + DELTA])


@fixture(name="test_patch_no_z")
def fixture_test_patch_no_z(test_points, tmp_path):
    info, data = DataChunk.create(test_points.ra, test_points.dec)

    path = tmp_path / "patch_no_z"
    writer = PatchWriter(path, chunk_info=info)
    writer.process_chunk(data)
    writer.close()

    return Patch(path)


@fixture(name="test_patch")
def fixture_test_patch(test_points, tmp_path):
    info, data = DataChunk.create(
        test_points.ra,
        test_points.dec,
        redshifts=np.arange(len(test_points)) % 2 + 0.5,
    )

    path = tmp_path / "patch"
    writer = PatchWriter(path, chunk_info=info)
    writer.process_chunk(data)
    writer.close()

    return Patch(path)


class TestBinnedTrees:
    def test_init_unbinned(self, test_patch_no_z):
        tree = trees.BinnedTrees.build(test_patch_no_z, None)
        repr(tree)

        assert tree.binning is None
        assert tree.binning_equal(None)
        assert not tree.is_binned()
        assert tree.num_bins is None

        iterator = iter(tree)
        assert_array_equal(next(iterator).data, tree.trees.data)
        assert_array_equal(next(iterator).data, tree.trees.data)

    def test_reload(self, test_patch):
        trees.BinnedTrees.build(test_patch, None)
        trees.BinnedTrees.build(test_patch, None)
        trees.BinnedTrees(test_patch)

    def test_init_binned(self, test_patch, test_patch_no_z):
        binning = Binning([0.0, 1.0, 2.0], closed="left")
        with raises(ValueError, match=".*no 'redshifts'.*"):
            trees.BinnedTrees.build(test_patch_no_z, binning)

        tree = trees.BinnedTrees.build(test_patch, binning)
        repr(tree)

        assert_array_equal(tree.binning.edges, binning.edges)
        assert tree.binning.closed == binning.closed
        assert tree.binning_equal(binning)
        assert not tree.binning_equal(binning[1:])
        assert tree.is_binned()
        assert tree.num_bins is len(binning)

        iterator = iter(tree)
        assert_array_equal(next(iterator).data, tree.trees[0].data)
        assert_array_equal(next(iterator).data, tree.trees[1].data)
        with raises(StopIteration):
            next(iterator)
