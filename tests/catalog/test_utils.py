import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import mark, raises

from yaw.catalog import utils


def test_groupby():
    n_items = 4
    items = np.arange(n_items)
    n_tile = 5
    array = np.tile(items, n_tile)

    for item, (key, data) in zip(items, utils.groupby(array, array)):
        assert item == key
        assert_array_equal(data, np.full(n_tile, item))


class TestPatchIDs:
    max_id = np.iinfo(np.dtype(utils.PatchIDs.itemtype)).max

    @mark.parametrize("patch_id", [0, max_id])
    def test_validate(self, patch_id):
        patch_ids = [patch_id, patch_id]
        utils.PatchIDs.validate(patch_id)
        utils.PatchIDs.validate(patch_ids)
        utils.PatchIDs.validate(np.array(patch_ids))

    def test_validate_wrong_dim(self):
        with raises(ValueError):
            utils.PatchIDs.validate([[0]])

    @mark.parametrize("patch_id", [-1, max_id + 1])
    def test_validate_out_of_range(self, patch_id):
        with raises(ValueError):
            utils.PatchIDs.validate(patch_id)

    @mark.parametrize("patch_ids", [0, max_id, [0, max_id]])
    def test_parse(self, patch_ids):
        try:
            num_expect = len(patch_ids)
        except TypeError:
            num_expect = 1
        utils.PatchIDs.parse(patch_ids, num_expect=num_expect)

        result = utils.PatchIDs.parse(patch_ids)
        assert result.dtype == np.dtype(utils.PatchIDs.itemtype)
        assert_array_equal(result, patch_ids)

    def test_parse_wrong_dim(self):
        with raises(ValueError):
            utils.PatchIDs.parse([[0]])

    @mark.parametrize("patch_id", [-1, max_id + 1])
    def test_parse_out_of_range(self, patch_id):
        with raises(ValueError):
            utils.PatchIDs.parse(patch_id)

    @mark.parametrize("patch_id", [0, [0]])
    def test_parse_wrong_length(self, patch_id):
        with raises(ValueError):
            utils.PatchIDs.parse(patch_id, num_expect=2)


def get_mock_data(dtype="f8"):
    data = np.arange(360.0, dtype=dtype)
    return dict(ra=data, dec=data, weights=data, redshifts=data)


class TestPatchData:
    drop_cases = [(), ("weights",), ("redshifts",), ("weights", "redshifts")]

    def build_dtype(self, columns):
        return np.dtype([(col, utils.PatchData.itemtype) for col in columns])

    @mark.parametrize("dtype", ["f8", "f4"])
    def test_from_columns(self, dtype):
        mock_data = get_mock_data(dtype)
        patch = utils.PatchData.from_columns(**mock_data)

        assert patch.dtype == self.build_dtype(mock_data)

        ra = np.deg2rad(mock_data["ra"])
        dec = np.deg2rad(mock_data["dec"])
        assert_array_almost_equal(ra, patch.coords.ra)
        assert_array_almost_equal(dec, patch.coords.dec)
        assert_array_almost_equal(mock_data["weights"], patch.weights)
        assert_array_almost_equal(mock_data["redshifts"], patch.redshifts)

        assert len(ra) == len(patch)

    @mark.parametrize("dtype", ["f8", "f4"])
    def test_from_columns_rad(self, dtype):
        mock_data = get_mock_data(dtype)
        patch = utils.PatchData.from_columns(**mock_data, degrees=False)

        assert_array_almost_equal(mock_data["ra"], patch.coords.ra)
        assert_array_almost_equal(mock_data["dec"], patch.coords.dec)

    @mark.parametrize("drop", drop_cases)
    def test_from_columns_optionals(self, drop):
        mock_data = get_mock_data()
        for key in drop:
            mock_data.pop(key)
        patch = utils.PatchData.from_columns(**mock_data)

        assert patch.dtype == self.build_dtype(mock_data)

        assert (patch.weights is None) is ("weights" in drop)
        assert (patch.redshifts is None) is ("redshifts" in drop)

    def test_repr(self):
        mock_patch = utils.PatchData.from_columns(**get_mock_data())
        repr(mock_patch)

    @mark.parametrize("drop", drop_cases)
    def test_write_read(self, drop, tmp_path):
        mock_data = get_mock_data()
        for key in drop:
            mock_data.pop(key)
        patch = utils.PatchData.from_columns(**mock_data)

        path = tmp_path / "patch.bin"
        patch.to_file(path)
        result = utils.PatchData.from_file(path)

        assert_array_equal(result.coords.ra, patch.coords.ra)
        assert_array_equal(result.coords.dec, patch.coords.dec)
        assert_array_equal(result.weights, patch.weights)
        assert_array_equal(result.redshifts, patch.redshifts)
