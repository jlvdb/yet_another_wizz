import numpy as np
import numpy.testing as npt
from pytest import mark

from yaw.core import utils


@mark.parametrize("axis", [None, 0, 1])
def test_outer_triu(axis):
    def ref(t1, t2, k=0, axis=None):
        return np.triu(np.outer(t1, t2), k).sum(axis)

    N = 10
    repeat = 10
    a = np.arange(1, N + 1)
    a2 = np.repeat(a, repeat).reshape((*a.shape, repeat))
    for k in range(-N, N):
        got = utils.outer_triu_sum(a2, a2+1, k=k, axis=axis)
        require = ref(a, a+1, k=k, axis=axis)
        require = np.repeat(require, repeat).reshape((*require.shape, repeat))
        npt.assert_equal(got, require)
