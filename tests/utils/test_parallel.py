from yaw.utils import parallel


class Dummy:
    pass


def dummy_task(*args, **kwargs):
    return args, kwargs


class TestMockComm:
    def test_Bcast(self):
        assert parallel.COMM.Bcast(Dummy) is Dummy

    def test_bcast(self):
        assert parallel.COMM.bcast(Dummy) is Dummy

    def test_Get_rank(self):
        assert parallel.COMM.Get_rank() == 0


def test_use_mpi():
    assert parallel.use_mpi() is False


def test_get_size():
    assert parallel.get_size() == 1


def test_on_root():
    assert parallel.on_root() is True


def test_on_worker():
    assert parallel.on_worker() is False


def test_ParallelJob():
    args = (1, 2, 3)
    kwargs = dict(a=1, b=2)

    job = parallel.ParallelJob(dummy_task, args, kwargs)
    run_args, run_kwargs = job((Dummy,))
    assert run_args == ((Dummy,), *args)
    assert run_kwargs == kwargs


def test_ParallelJob_unpack():
    args = (1, 2, 3)
    kwargs = dict(a=1, b=2)

    job = parallel.ParallelJob(dummy_task, args, kwargs, unpack=True)
    run_args, run_kwargs = job((Dummy,))
    assert run_args == (Dummy, *args)
    assert run_kwargs == kwargs


def test_iter_unordered():
    job = parallel.ParallelJob(dummy_task, (), {})
    args = range(10)

    for arg, (run_args, _) in zip(args, parallel.iter_unordered(job, args)):
        assert run_args == (arg,)
