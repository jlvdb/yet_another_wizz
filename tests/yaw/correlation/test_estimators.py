import numpy as np
from pytest import mark, raises, warns

from yaw.correlation import estimators


class TestCts:
    def test_set_squashing(self):
        assert len({estimators.CtsDR(), estimators.CtsRD(), estimators.CtsMix()}) == 2

    def test_equivalents(self):
        assert estimators.CtsDR() == estimators.CtsMix()
        assert estimators.CtsRD() == estimators.CtsMix()

    def test_repr_checks(self):
        repr(estimators.CtsMix())
        str(estimators.CtsMix())


@mark.parametrize(
    "code,counts",
    [
        ("dd", estimators.CtsDD),
        ("dr", estimators.CtsDR),
        ("rd", estimators.CtsRD),
        ("rr", estimators.CtsRR),
    ],
)
def test_cts_from_code(code, counts):
    assert counts() == estimators.cts_from_code(code)
    with raises(ValueError):
        estimators.cts_from_code("undefined symbol")


class TestPeeblesHauser:
    def test_eval(self):
        assert estimators.PeeblesHauser.eval(dd=1, rr=1) == 0
        with warns(UserWarning):
            estimators.PeeblesHauser.eval(dd=np.float_(1), rr=np.float_(0))


class TestDavisPeebles:
    def test_eval(self):
        assert estimators.DavisPeebles.eval(dd=1, dr_rd=1) == 0
        with warns(UserWarning):
            estimators.DavisPeebles.eval(dd=np.float_(1), dr_rd=np.float_(0))


class TestHamilton:
    def test_eval(self):
        assert estimators.Hamilton.eval(dd=1, dr=1, rr=1) == 0
        assert estimators.Hamilton.eval(dd=1, dr=1, rd=1, rr=1) == 0
        with warns(UserWarning):
            estimators.PeeblesHauser.eval(
                dd=np.float_(1), dr=np.float_(1), rr=np.float_(0)
            )


class TestLandySzalay:
    def test_eval(self):
        assert estimators.LandySzalay.eval(dd=1, dr=1, rr=1) == 0
        assert estimators.LandySzalay.eval(dd=1, dr=1, rd=1, rr=1) == 0
        with warns(UserWarning):
            estimators.PeeblesHauser.eval(
                dd=np.float_(1), dr=np.float_(1), rr=np.float_(0)
            )
