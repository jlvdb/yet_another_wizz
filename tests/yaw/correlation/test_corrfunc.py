import numpy as np
import numpy.testing as npt
import pandas as pd
from pytest import fixture, mark, raises

from yaw import examples
from yaw.correlation import corrfuncs, estimators


@fixture
def binning():
    return pd.IntervalIndex.from_breaks([0.0, 0.5, 1.0, 1.5])


@fixture
def test_data_samples(binning):
    """
    Construct some fake data with value 4 and samples that yield a variance of
    4 2/3 for bootstrapping and 24 for jackknifing. The covariance matrix is
    constant.
    """
    n_bins, n_samples = len(binning), 7
    data = np.full(n_bins, 4)
    samples = np.repeat(np.arange(1, 8), n_bins).reshape((n_samples, n_bins))
    return binning, data, samples


@fixture
def stats_jackknife(binning):
    var = 24.0
    err = np.full(len(binning), np.sqrt(var))
    cov = np.full((len(binning), len(binning)), var)
    cor = np.ones_like(cov)
    return err, cov, cor


@fixture
def stats_bootstrap(binning):
    var = 4.0 + 2.0 / 3.0
    err = np.full(len(binning), np.sqrt(var))
    cov = np.full((len(binning), len(binning)), var)
    cor = np.ones_like(cov)
    return err, cov, cor


@fixture
def correlation_data(test_data_samples):
    binning, data, samples = test_data_samples
    return corrfuncs.CorrData(binning, data, samples, method="jackknife", info="info")


def text_grep(path, pattern, sub):
    with open(path) as f:
        lines = [line.replace(pattern, sub) for line in f.readlines()]
    with open(path, "w") as f:
        for line in lines:
            f.write(line)


class TestCorrData:
    def test_stats(self, test_data_samples, stats_jackknife, stats_bootstrap):
        binning, data, samples = test_data_samples
        # jackknife
        err, cov, cor = stats_jackknife
        cd = corrfuncs.CorrData(binning, data, samples, method="jackknife")
        npt.assert_allclose(cd.error, err)
        npt.assert_allclose(cd.covariance, cov)
        npt.assert_allclose(cd.get_correlation().to_numpy(), cor)
        # bootstrap
        err, cov, cor = stats_bootstrap
        cd = corrfuncs.CorrData(binning, data, samples, method="bootstrap")
        npt.assert_allclose(cd.error, err)
        npt.assert_allclose(cd.covariance, cov)
        npt.assert_allclose(cd.get_correlation().to_numpy(), cor)

    def test_io(self, tmp_path_factory, correlation_data):
        path = tmp_path_factory.mktemp("corr_data_file") / "testfiles"
        correlation_data.to_files(path)
        restored = correlation_data.from_files(path)
        npt.assert_allclose(correlation_data.data, restored.data)
        npt.assert_allclose(correlation_data.samples, restored.samples)
        assert correlation_data.info == restored.info
        # break the data file
        text_grep(f"{path}.smp", correlation_data.method[:4], "udef")
        with raises(ValueError):
            correlation_data.from_files(path)
        text_grep(f"{path}.smp", "udef", correlation_data.method[:4])
        text_grep(f"{path}.smp", "z_low", "wrong_key")
        with raises(ValueError):
            correlation_data.from_files(path)

    @mark.parametrize("flags", [False, True])
    def test_plot(self, correlation_data, flags):
        correlation_data.plot(error_bars=flags, zero_line=flags, scale_by_dz=flags)

    @mark.parametrize("redshift", [False, True])
    def test_plot_corr(self, correlation_data, redshift):
        correlation_data.plot_corr(redshift=redshift)


def test_check_mergable():
    corrfuncs.check_mergable([examples.w_sp, examples.w_sp])
    # different pair counts
    with raises(ValueError, match="incompatible"):
        corrfuncs.check_mergable([examples.w_sp, examples.w_ss])


class TestCorrFunc:
    def test_init(self):
        pairs = dict(
            dd=examples.w_ss.dd,
            dr=examples.w_ss.dr,
            rd=examples.w_ss.dr,  # duplicated
            rr=examples.w_ss.rr,
        )
        assert examples.w_ss == corrfuncs.CorrFunc(
            dd=pairs["dd"], dr=pairs["dr"], rr=pairs["rr"]
        )
        with raises(ValueError, match="required"):
            corrfuncs.CorrFunc(dd=pairs["dd"])
        for name, pair in pairs.items():
            if name == "dd":
                continue
            corrfuncs.CorrFunc(dd=pairs["dd"], **{name: pair})
        with raises(ValueError):
            corrfuncs.CorrFunc(dd=pairs["dd"], dr=pairs["dr"].bins[1:])
        with raises(ValueError):
            corrfuncs.CorrFunc(dd=pairs["dd"], dr=pairs["dr"].patches[1:])

    def test_repr(self):
        repr(examples.w_ss)

    def test_eq(self):
        assert examples.w_sp == examples.w_sp
        assert examples.w_sp != examples.w_ss
        assert examples.w_sp != 1

    def test_add(self):
        summed = examples.w_ss + examples.w_ss
        assert summed.dd == examples.w_ss.dd + examples.w_ss.dd
        assert summed.dr == examples.w_ss.dr + examples.w_ss.dr
        assert summed.rr == examples.w_ss.rr + examples.w_ss.rr
        with raises(ValueError, match="not set"):
            new = corrfuncs.CorrFunc(dd=summed.dd, dr=summed.dr)
            summed + new
        with raises(TypeError):
            summed + 1

    def test_radd(self):
        assert (0 + examples.w_sp) == examples.w_sp
        assert sum([examples.w_sp]) == examples.w_sp
        with raises(TypeError):
            1 + examples.w_sp

    def test_mul(self):
        mul = examples.w_sp * 2
        assert mul == examples.w_sp + examples.w_sp
        assert mul.dd.total == examples.w_sp.dd.total
        assert mul.dd.count == examples.w_sp.dd.count * 2
        with raises(TypeError):
            examples.w_sp * True

    def test_auto(self):
        assert examples.w_sp.auto == examples.w_sp.dd.count.auto

    def test_n_patches(self):
        assert examples.w_sp.n_patches == examples.w_sp.dd.count.n_patches

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_bins(self, items):
        subset = examples.w_sp.bins[items]
        assert subset == corrfuncs.CorrFunc(
            dd=examples.w_sp.dd.bins[items], dr=examples.w_sp.dr.bins[items]
        )

    @mark.parametrize("items", [1, 2, slice(0, 2), [1, 2]])
    def test_patches(self, items):
        subset = examples.w_sp.patches[items]
        assert subset == corrfuncs.CorrFunc(
            dd=examples.w_sp.dd.patches[items], dr=examples.w_sp.dr.patches[items]
        )

    def test_get_binning(self):
        assert (examples.w_sp.get_binning() == examples.w_sp.dd.get_binning()).all()

    def test_is_compatible(self):
        examples.w_sp.is_compatible(examples.w_sp)
        with raises(ValueError):
            assert not examples.w_sp.is_compatible(examples.w_sp.bins[1:])
            assert examples.w_sp.is_compatible(examples.w_sp.bins[1:], require=True)
        with raises(ValueError):
            assert not examples.w_sp.is_compatible(examples.w_sp.patches[1:])
            assert examples.w_sp.is_compatible(examples.w_sp.patches[1:], require=True)

    def test_estimators(self):
        pairs = dict(
            dd=examples.w_ss.dd,
            dr=examples.w_ss.dr,
            rd=examples.w_ss.dr,  # duplicated
            rr=examples.w_ss.rr,
        )
        cf = corrfuncs.CorrFunc(**pairs).estimators
        assert set(cf.keys()) == {"HM", "LS", "DP", "PH"}
        cf = corrfuncs.CorrFunc(
            dd=pairs["dd"], dr=pairs["dr"], rr=pairs["rr"]
        ).estimators
        assert set(cf.keys()) == {"HM", "LS", "DP", "PH"}
        cf = corrfuncs.CorrFunc(
            dd=pairs["dd"], rd=pairs["rd"], rr=pairs["rr"]
        ).estimators
        assert set(cf.keys()) == {"DP", "PH"}

        cf = corrfuncs.CorrFunc(
            dd=pairs["dd"], dr=pairs["dr"], rd=pairs["rd"]
        ).estimators
        assert set(cf.keys()) == {"DP"}
        cf = corrfuncs.CorrFunc(dd=pairs["dd"], rd=pairs["rd"]).estimators
        assert set(cf.keys()) == {"DP"}
        cf = corrfuncs.CorrFunc(dd=pairs["dd"], dr=pairs["dr"]).estimators
        assert set(cf.keys()) == {"DP"}

        cf = corrfuncs.CorrFunc(dd=pairs["dd"], rr=pairs["rr"]).estimators
        assert set(cf.keys()) == {"PH"}

    def test__check_and_select_estimator(self):
        pairs = dict(
            dd=examples.w_ss.dd,
            dr=examples.w_ss.dr,
            rd=examples.w_ss.dr,  # duplicated
            rr=examples.w_ss.rr,
        )

        cf = corrfuncs.CorrFunc(**pairs)
        with raises(ValueError, match="invalid"):
            cf._check_and_select_estimator("undefined")
        assert cf._check_and_select_estimator() == estimators.LandySzalay
        for est in (
            estimators.LandySzalay,
            estimators.Hamilton,
            estimators.DavisPeebles,
            estimators.PeeblesHauser,
        ):
            assert cf._check_and_select_estimator(est.short) == est

        for pair in ("dr", "rd"):
            cf = corrfuncs.CorrFunc(dd=pairs["dd"], **{pair: pairs[pair]})
            assert cf._check_and_select_estimator() == estimators.DavisPeebles
            for est in (estimators.DavisPeebles,):
                assert cf._check_and_select_estimator(est.short) == est
            for est in (
                estimators.LandySzalay,
                estimators.Hamilton,
                estimators.PeeblesHauser,
            ):
                with raises(estimators.EstimatorError, match="requires"):
                    cf._check_and_select_estimator(est.short)

        cf = corrfuncs.CorrFunc(dd=pairs["dd"], rr=pairs["rr"])
        assert cf._check_and_select_estimator() == estimators.PeeblesHauser
        for est in (estimators.PeeblesHauser,):
            assert cf._check_and_select_estimator(est.short) == est
        for est in (
            estimators.LandySzalay,
            estimators.Hamilton,
            estimators.DavisPeebles,
        ):
            with raises(estimators.EstimatorError, match="requires"):
                cf._check_and_select_estimator(est.short)

    def test__getattr_from_cts(self):
        pairs = dict(
            dd=examples.w_ss.dd,
            dr=examples.w_ss.dr,
            rd=examples.w_ss.dr,  # duplicated
            rr=examples.w_ss.rr,
        )
        with_rd = corrfuncs.CorrFunc(**pairs)
        rd = pairs.pop("rd")
        without_rd = corrfuncs.CorrFunc(**pairs)
        # CtsDD
        assert with_rd.dd == with_rd._getattr_from_cts(estimators.CtsDD())
        assert with_rd.dd == without_rd._getattr_from_cts(estimators.CtsDD())
        # CtsDR
        assert with_rd.dr == with_rd._getattr_from_cts(estimators.CtsDR())
        assert with_rd.dr == without_rd._getattr_from_cts(estimators.CtsDR())
        # CtsRD
        assert with_rd.rd == with_rd._getattr_from_cts(estimators.CtsRD())
        assert without_rd._getattr_from_cts(estimators.CtsRD()) is None
        # CtsRR
        assert with_rd.rr == with_rd._getattr_from_cts(estimators.CtsRR())
        assert with_rd.rr == without_rd._getattr_from_cts(estimators.CtsRR())
        # CtsMix
        assert with_rd.dr == with_rd._getattr_from_cts(estimators.CtsMix())
        assert with_rd.dr == without_rd._getattr_from_cts(estimators.CtsMix())
        without_dr = corrfuncs.CorrFunc(**pairs, rd=rd)
        assert without_dr.rd == without_dr._getattr_from_cts(estimators.CtsMix())

    @mark.skip
    def test_sample(self):
        assert 0

    @mark.parametrize("cf", [examples.w_sp, examples.w_ss, examples.w_pp])
    def test_io(self, tmp_path_factory, cf):
        path = tmp_path_factory.mktemp("corr_func_file") / "testfile.hdf"
        cf.to_file(path)
        assert corrfuncs.CorrFunc.from_file(path) == cf

    @mark.skip
    def test_concatenate_patches(self):
        assert 0

    @mark.skip
    def test_concatenate_bins(self):
        assert 0


@mark.skip
def test__create_dummy_counts(self):
    assert 0


@mark.skip
def test_add_corrfuncs(self):
    assert 0


@mark.skip
def test__check_patch_centers(self):
    assert 0


@mark.skip
def test_autocorrelate(self):
    assert 0


@mark.skip
def test_crosscorrelate(self):
    assert 0
