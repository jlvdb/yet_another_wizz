import inspect
import json
import multiprocessing
import operator
import sys

import numpy as np
from astropy.cosmology import FLRW, default_cosmology
from matplotlib import colors
from matplotlib import pyplot as plt

from .spatial import FastSeparation2Angle


def load_json(path):
    with open(path) as f:
        data_dict = json.load(f)
        # convert lists to numpy arrays
        for key, value in data_dict.items():
            if type(value) is list:
                data_dict[key] = np.array(value)
    return data_dict


def dump_json(data, path, preview=False):
    kwargs = dict(indent=4, default=operator.methodcaller("tolist"))
    if preview:
        print(json.dumps(data, **kwargs))
    else:
        with open(path, "w") as f:
            json.dump(data, f, **kwargs)


class ThreadHelper(object):
    """
    ThreadHelper(threads, n_items)

    Helper class to apply a series of arguments to a function using
    multiprocessing.Pool.

    Parameters
    ----------
    threads : int
        Number of parallel threads for the pool.
    n_items : int
        Number items to be processed by the threads.
    """

    threads = multiprocessing.cpu_count()

    def __init__(self, n_items, threads=None):
        self._n_items = n_items
        if threads is not None:
            self.threads = threads
        self.args = []

    def __len__(self):
        """
        Returns the number of expected function arguments.
        """
        return self._n_items

    def add_constant(self, value):
        """
        add_constant(value)

        Append a constant argument that will be repeated for each thread.

        Parameters
        ----------
        value : any
            Any object that can be serialized.
        """
        self.args.append([value] * self._n_items)

    def add_iterable(self, iterable, no_checks=False):
        """
        add_iterable(iterable, no_checks=False)

        Append a constant argument that will be repeated for each thread.

        Parameters
        ----------
        iterable : iterable
            Any iterator that yields objects that can be serialized.
        no_checks : bool
            Whether the iterable should be checked if it supports iteration.
        """
        if not hasattr(iterable, "__iter__") and not no_checks:
            raise TypeError("object is not iterable: %s" % iterable)
        self.args.append(iterable)

    def map(self, callable):
        """
        map(callable)

        Apply the accumulated arguments to a function in a pool of threads.
        The threads are blocking until all results are received.

        Parameters
        ----------
        callable : callable
            Callable object with the correct number of arguements.

        Returns:
        --------
        results: list
            Ordered list of return values of the called object per thread.
        """
        if self.threads > 1:
            with multiprocessing.Pool(self.threads) as pool:
                results = pool.starmap(callable, zip(*self.args))
        else:  # mimic the behaviour of pool.starmap() with map()
            results = list(map(callable, *self.args))
        return results


class BaseClass(object):

    cosmology = default_cosmology.get()
    _verbose = True

    def _printMessage(self, message):
        if self._verbose:
            classname = self.__class__.__name__
            methodname = inspect.stack()[1].function
            sys.stdout.write("%s::%s - %s" % (classname, methodname, message))
            sys.stdout.flush()

    def _throwException(self, message, exception_type):
        classname = self.__class__.__name__
        methodname = inspect.stack()[1].function
        exception = exception_type(message)
        exceptionname = exception.__class__.__name__
        sys.stdout.write(
            "%s::%s - %s: %s\n" % (
                classname, methodname, exceptionname, message))
        sys.stdout.flush()
        raise exception

    def setCosmology(self, cosmology):
        if type(cosmology) is str:
            self.cosmology = \
                default_cosmology.get_cosmology_from_string(cosmology)
        elif not issubclass(type(cosmology), FLRW):
            self._throwException(
                "cosmology must be subclass of type %s" % FLRW, TypeError)
        else:
            self.cosmology = cosmology
