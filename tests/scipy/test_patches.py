import itertools

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pytest import fixture, raises

from yaw.scipy import patches


def test_patch_id_from_path():
    fpath = "my/cache/dir/big_file_1.feather"
    assert patches.patch_id_from_path(fpath) == 1


def test_patch_id_from_path_no_feather():
    fpath = "my/cache/dir/big_file_1.fits"
    with raises(patches.NotAPatchFileError):
        patches.patch_id_from_path(fpath)


@fixture
def mock_data():
    ra_dec = []
    for ra, dec in itertools.product(range(0, 355), range(-89, 90)):
        ra_dec.append([ra, dec])
    ra_dec = np.array(ra_dec, dtype=np.float_)
    df = pd.DataFrame(dict(
        ra=ra_dec[:, 0], dec=ra_dec[:, 1],
        weights=np.ones(len(ra_dec)),
        redshift=np.linspace(0.0, 2.0, len(ra_dec))))
    return df


@fixture
def mock_data_rad(mock_data):
    data = mock_data.copy()
    for key in ("ra", "dec"):
        data[key] = np.deg2rad(mock_data[key])
    return data


@fixture
def mock_patch(mock_data):
    return patches.PatchCatalog(0, mock_data, degrees=True)


@fixture
def patch_cached(mock_data, tmp_path):
    path = str(tmp_path / "test_1.feather")
    patch = patches.PatchCatalog(0, mock_data, degrees=True, cachefile=path)
    patch.unload()
    yield patch


class TestPatchCatalog:

    def test_init(self, mock_data):
        """attributes .data"""
        patch = patches.PatchCatalog(0, mock_data, degrees=False)
        assert patch.id == 0
        pdt.assert_frame_equal(patch.data, mock_data)
        # required columns
        for col in ("ra", "dec"):
            with raises(KeyError):
                patches.PatchCatalog(0, mock_data.drop(columns=col))
        # extra column
        with raises(KeyError):
            patches.PatchCatalog(0, mock_data.rename(columns={"redshift": "z"}))

    def test_cache_file(self, mock_data, mock_data_rad, tmp_path):
        patch_id = 10

        # create with rad data
        cachefile = tmp_path / f"test_rad_{patch_id}.feather"
        patch = patches.PatchCatalog(
            patch_id, mock_data_rad, degrees=False, cachefile=cachefile)
        assert cachefile.exists()
        pdt.assert_frame_equal(pd.read_feather(cachefile), patch.data)
        # reload file
        patch = patches.PatchCatalog.from_cached(str(cachefile))
        pdt.assert_frame_equal(mock_data_rad, patch.data)
        assert patch.id == patch_id

        # create with degrees data
        cachefile = tmp_path / f"test_deg_{patch_id}.feather"
        patch = patches.PatchCatalog(
            patch_id, mock_data, degrees=True, cachefile=cachefile)
        assert cachefile.exists()
        pdt.assert_frame_equal(pd.read_feather(cachefile), patch.data)
        # reload file
        patch = patches.PatchCatalog.from_cached(str(cachefile))
        pdt.assert_frame_equal(mock_data_rad, patch.data)
        assert patch.id == patch_id

    def test_loading(self, patch_cached):
        """is_loaded, require_loaded, load, unload"""
        # is unloaded
        assert not patch_cached.is_loaded()
        assert patch_cached._data is None
        with raises(patches.CachingError, match=".*not loaded.*"):
            patch_cached.require_loaded()
        # is loaded
        patch_cached.load()
        assert patch_cached.is_loaded()
        assert isinstance(patch_cached._data, pd.DataFrame)

        # remove cache file
        patch_cached.cachefile = None
        with raises(patches.CachingError, match=".*no datapath.*"):
            patch_cached.unload()
        # manually unload
        patch_cached._data = None
        with raises(patches.CachingError, match=".*no datapath.*"):
            patch_cached.load()

    def test_len(self, mock_data, mock_patch, patch_cached):
        assert len(mock_data) == len(mock_patch) == len(patch_cached)

    def test_positional(self, mock_data, mock_data_rad, patch_cached):
        """attributes .pos, .ra, .dec"""
        patch = patches.PatchCatalog(0, mock_data, degrees=True)
        npt.assert_equal(patch.ra, mock_data_rad["ra"])
        npt.assert_equal(patch.dec, mock_data_rad["dec"])
        # check everything works if there is no conversion
        patch = patches.PatchCatalog(0, mock_data_rad, degrees=False)
        npt.assert_equal(patch.ra, mock_data_rad["ra"])
        npt.assert_equal(patch.dec, mock_data_rad["dec"])
        # unloaded
        for attr in ("pos", "ra", "dec"):
            with raises(patches.CachingError):
                getattr(patch_cached, attr)

    def test_redshifts(self, mock_data, mock_patch, patch_cached):
        assert mock_patch.has_redshifts()
        npt.assert_equal(mock_patch.redshifts, mock_data["redshift"])
        # drop redshifts
        patch = patches.PatchCatalog(
            0, mock_data.drop(columns="redshift"), degrees=True)
        assert not patch.has_redshifts()
        # unloaded
        with raises(patches.CachingError):
            patch_cached.redshifts

    def test_weights(self, mock_data, mock_patch, patch_cached):
        assert mock_patch.has_weights()
        npt.assert_equal(mock_patch.weights, mock_data["weights"])
        # drop weights
        patch = patches.PatchCatalog(
            0, mock_data.drop(columns="weights"), degrees=True)
        assert not patch.has_weights()
        # TODO: unloaded
        with raises(patches.CachingError):
            patch_cached.weights

    def test_total(self, mock_data, mock_patch, patch_cached):
        assert mock_patch.total == mock_data["weights"].sum()
        # drop weights
        patch = patches.PatchCatalog(
            0, mock_data.drop(columns="weights"), degrees=True)
        assert len(patch) == len(mock_data)
        # unloaded
        patch_cached.total == mock_data["weights"].sum()

    def test_iter_bins(self, mock_patch, mock_data):
        # check the redshift intervals
        zbins = [0.1, 0.5, 1.0]
        for i, (intv, patch) in enumerate(mock_patch.iter_bins(zbins)):
            assert isinstance(intv, pd.Interval)
            assert isinstance(patch, patches.PatchCatalog)
            assert patch.redshifts.min() >= intv.left == zbins[i]
            assert patch.redshifts.max() <= intv.right == zbins[i+1]
            assert patch.ra.max() < 2*np.pi  # cheap way to rule out degrees

        # no redshifts
        patch_noz = patches.PatchCatalog(0, mock_data.drop(columns="redshift"))
        with raises(ValueError):
            next(patch_noz.iter_bins(zbins))
        for intv, patch in patch_noz.iter_bins(zbins, allow_no_redshift=True):
            assert len(patch) == len(patch_noz)

    def test_get_tree(self, mock_patch):
        """just check if the tree can be initialised"""
        tree = mock_patch.get_tree(leafsize=10)
        npt.assert_equal(tree.weights, mock_patch.weights)


"""
center
radius
"""
