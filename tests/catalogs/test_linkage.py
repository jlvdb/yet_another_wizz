import numpy as np
import numpy.testing as npt
from pytest import fixture

from yaw.catalogs import PatchLinkage
from yaw.catalogs.linkage import LINK_ZMIN
from yaw.config import Configuration
from yaw.core.coordinates import CoordSky, DistSky


class MockCatalog:
    """
    This is a mock catalog with 2x3 patches, centered at (RA/Dec) = (n, -1) and
    (RA/Dec) = (n, 1), with n=0,2,4 in degrees.
           +-+-+-+
           |3|4|5|
    Dec=0  +-+-+-+  (indices shown 0..5)
           |0|1|2|
           +-+-+-+

    For the case of no overlap between diagonal neighbours (radius < 1.0), the
    following linkange pattern should arise (mirror for cross-correlation):
        0 1 2 3 4 5    
       +-+-+-+-+-+-+
    0  |#|x| |x| | |
       +-+-+-+-+-+-+
    1  | |#|x| |x| |
       +-+-+-+-+-+-+
    2  | | |#| | |x|
       +-+-+-+-+-+-+
    3  | | | |#|x| |
       +-+-+-+-+-+-+
    4  | | | | |#|x|
       +-+-+-+-+-+-+
    5  | | | | | |#|
       +-+-+-+-+-+-+
    """
    centers=CoordSky(
        ra= np.deg2rad([ 0, 2, 4, 0, 2, 4]),
        dec=np.deg2rad([-1,-1,-1, 1, 1, 1]))
    radii=DistSky(np.deg2rad([1.00001] * 6))  # > 1 necessary
    n_patches=6

    def __getitem__(self, patch_id):
        return patch_id


@fixture
def config():
    return Configuration.create(
        rmin=0.0000001, rmax=0.000001,  # some non-zero small value
        zbins=[0.01, 0.1])


@fixture
def catalog():
    return MockCatalog()


@fixture
def linkage(config, catalog):
    return PatchLinkage.from_setup(config, catalog)


@fixture
def pairs():
    return [
        (0,0), (0,1), (0,3),
        (1,0), (1,1), (1,2), (1,4),
        (2,1), (2,2), (2,5),
        (3,0), (3,3), (3,4),
        (4,1), (4,3), (4,4), (4,5),
        (5,2), (5,4), (5,5)]


@fixture
def pairs_auto():
    return [
        (0,0), (0,1), (0,3),
        (1,1), (1,2), (1,4),
        (2,2), (2,5),
        (3,3), (3,4),
        (4,4), (4,5),
        (5,5)]


@fixture
def pairs_nocross():
    return [(i,i) for i in range(6)]


@fixture
def mask():
    return np.ones((6, 6), dtype="bool")


@fixture
def mask_auto(mask):
    return np.triu(mask)


@fixture
def matrix():
    return np.array([
        [1,1,0,1,0,0],
        [1,1,1,0,1,0],
        [0,1,1,0,0,1],
        [1,0,0,1,1,0],
        [0,1,0,1,1,1],
        [0,0,1,0,1,1]],
        dtype="bool")


@fixture
def matrix_auto(matrix):
    return np.triu(matrix)


@fixture
def matrix_mask_nocross():
    return np.eye(6, dtype="bool")


class TestPatchLinkage:

    def test_from_setup(self, linkage, pairs):
        assert set(linkage.pairs) == set(pairs)
        repr(linkage)

    def test_len(self, linkage, pairs):
        assert len(linkage) == len(pairs)

    def test_n_patches(self, linkage):
        assert linkage.n_patches == 6

    def test_density(self, linkage, pairs):
        assert linkage.density == len(pairs)/36.

    def test_get_pairs(self, linkage, pairs, pairs_auto, pairs_nocross):
        assert set(linkage.get_pairs(auto=False)) == set(pairs)
        assert set(linkage.get_pairs(auto=True)) == set(pairs_auto)
        assert set(linkage.get_pairs(auto=True, crosspatch=False)) == set(pairs_nocross)

    def test_get_matrix(self, linkage, catalog, matrix, matrix_auto, matrix_mask_nocross):
        # cross
        npt.assert_equal(
            linkage.get_matrix(catalog, catalog),
            matrix)
        # auto
        npt.assert_equal(
            linkage.get_matrix(catalog),
            matrix_auto)
        npt.assert_equal(
            linkage.get_matrix(catalog, None),
            matrix_auto)
        # diag
        npt.assert_equal(
            linkage.get_matrix(catalog, catalog, crosspatch=False),
            matrix_mask_nocross)
        npt.assert_equal(
            linkage.get_matrix(catalog, None, crosspatch=False),
            matrix_mask_nocross)

    def test_get_mask(self, linkage, catalog, mask, mask_auto, matrix_mask_nocross):
        # cross
        npt.assert_equal(
            linkage.get_mask(catalog, catalog),
            mask)
        # auto
        npt.assert_equal(
            linkage.get_mask(catalog),
            mask_auto)
        npt.assert_equal(
            linkage.get_mask(catalog, None),
            mask_auto)
        # diag
        npt.assert_equal(
            linkage.get_mask(catalog, catalog, crosspatch=False),
            matrix_mask_nocross)
        npt.assert_equal(
            linkage.get_mask(catalog, None, crosspatch=False),
            matrix_mask_nocross)

    def test_query_radius(self, catalog):
        config = Configuration.create(
            rmin=1, rmax=8980,  # should barely exclude the four longest pairs
            zbins=[LINK_ZMIN, 0.1], crosspatch=False)
        assert len(PatchLinkage.from_setup(config, catalog)) == 20
        # distance between patches 0 and 5: approx. sqrt(20), with patch radius
        # 1 -> maximum query radius = 0.1561 rad to create overlap
        config = Configuration.create(
            rmin=1, rmax=8980,  # should barely exclude the four longest pairs
            zbins=[LINK_ZMIN, 0.1])
        assert len(PatchLinkage.from_setup(config, catalog)) == 32
        config = Configuration.create(
            rmin=1, rmax=8990,  # should be just enough to include all
            zbins=[LINK_ZMIN, 0.1])
        assert len(PatchLinkage.from_setup(config, catalog)) == 36

    def test_get_patches(self, linkage, catalog, pairs, pairs_auto, pairs_nocross):
        # cross
        result1, result2 = [], []
        for p1, p2 in pairs:
            result1.append(p1)
            result2.append(p2)
        patches1, patches2 = linkage.get_patches(catalog, catalog)
        assert tuple(patches1) == tuple(result1)
        assert tuple(patches2) == tuple(result2)
        # auto
        result1, result2 = [], []
        for p1, p2 in pairs_auto:
            result1.append(p1)
            result2.append(p2)
        patches1, patches2 = linkage.get_patches(catalog)
        assert tuple(patches1) == tuple(result1)
        assert tuple(patches2) == tuple(result2)
        patches1, patches2 = linkage.get_patches(catalog, None)
        assert tuple(patches1) == tuple(result1)
        assert tuple(patches2) == tuple(result2)
        # diag
        result1, result2 = [], []
        for p1, p2 in pairs_nocross:
            result1.append(p1)
            result2.append(p2)
        patches1, patches2 = linkage.get_patches(catalog, catalog, crosspatch=False)
        assert tuple(patches1) == tuple(result1)
        assert tuple(patches2) == tuple(result2)
        patches1, patches2 = linkage.get_patches(catalog, None, crosspatch=False)
        assert tuple(patches1) == tuple(result1)
        assert tuple(patches2) == tuple(result2)
