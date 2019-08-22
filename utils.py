import inspect
import multiprocessing
import sys

import numpy as np
import pandas as pd
import stomp
from astropy.cosmology import FLRW, default_cosmology
from matplotlib import colors
from matplotlib import pyplot as plt

from .spatial import FastSeparation2Angle


def region_color_map(base_cmap, n_regions, cycle_length=10):
    """
    region_color_map(base_cmap, n_regions, cycle_length=10)

    Create a cyclic discrete matplotlib colour map useful to plot STOMP mask
    regions.

    Parameters
    ----------
    base_cmap : str
        Name of a builtin matplotlib colormap.
    n_regions : int
        Number of expected STOMP mask regions.
    cycle_length : int
        Length of the colour cycle.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colour map to be used with matplotlib.
    """
    basemap = plt.cm.get_cmap(base_cmap)  # get one of the builtin maps
    # draw colours from the map and repeat the cycle
    repeats = n_regions // cycle_length + 1
    color_list = np.asarray(basemap(np.linspace(0, 1, cycle_length)))
    color_list = np.concatenate([color_list] * repeats, axis=0)
    color_list = color_list[:n_regions]
    # create the new color map from the cycle
    new_name = basemap.name + "_c%d_r%d" % (cycle_length, n_regions)
    cmap = colors.LinearSegmentedColormap.from_list(
        new_name, color_list, n_regions)
    return cmap


def measure_region_area(mask_path, n_regions, region_idx):
    """
    measure_region_area(mask_path, n_regions, region_idx)

    Measure the area of the i-th region of a given STOMP mask.

    Parameters
    ----------
    mask_path : str
        File path of STOMP mask.
    n_regions : int
        Number of STOMP mask regions.
    region_idx : int
        i-th region index, must be in range [0, n_regions).

    Returns
    -------
    area : float
        Area of the i-th region in square degrees.
    """
    # initialize the region mask
    master_mask = stomp.Map(mask_path)
    master_mask.InitializeRegions(n_regions)
    region_mask = stomp.Map()
    master_mask.RegionOnlyMap(region_idx, region_mask)
    # get the region area
    area = region_mask.Area()
    del master_mask, region_mask
    return area


def regionize_data(mask_path, n_regions, data):
    """
    regionize_data(mask_path, n_regions, data)

    Assign data objects to the regions of a given STOMP mask based on their
    right ascension / declination.

    Parameters
    ----------
    mask_path : str
        File path of STOMP mask.
    n_regions : int
        Number of STOMP mask regions.
    data : array_like or pandas.DataFrame
        Must of array of shape (Nx2) with N entries of type (RA, DEC) or a
        pandas DataFrame with columns RA and DEC, angles given in degrees.

    Returns
    -------
    area : array
        List of N region indices in range [0, n_regions) indicating the region
        membership of the objects. If index == -1, the object falls outside the
        mask footprint.
    """
    try:  # if data is a pandas.DataFrame
        RA = data.RA
        DEC = data.DEC
    except AttributeError:  # if data is a Nx2 np.ndarray
        RA, DEC = data.T
    # load mask
    stomp_mask = stomp.Map(mask_path)
    stomp_mask.InitializeRegions(n_regions)
    # check for every object in which stomp region it falls
    region_idx = np.empty(len(RA), dtype=np.int16)
    for i, (ra, dec) in enumerate(zip(RA, DEC)):
        ang = stomp.AngularCoordinate(
            ra, dec, stomp.AngularCoordinate.Equatorial)
        # first we must check whether the object falls into the mask at all,
        # since regions may extend over the mask bounds
        if stomp_mask.Contains(ang):
            region_idx[i] = stomp_mask.FindRegion(ang)
        else:
            region_idx[i] = -1
    del stomp_mask, ang
    return region_idx


def generate_randoms(mask_path, n_randoms):
    """
    generate_randoms(mask_path, n_randoms)

    Generate a sample of uniformly distributed objects on a STOMP mask.

    Parameters
    ----------
    mask_path : str
        File path of STOMP mask.
    n_randoms : int
        Number of random points to generate.

    Returns
    -------
    area : pandas.DataFrame
        Data frame with columns RA (right ascension, in degrees) and DEC
        (declination, in degrees) of the random objects.
    """
    # use the stomp methods to generate uniform random points
    stomp_map = stomp.Map(mask_path)
    random_vector = stomp.AngularVector()
    stomp_map.GenerateRandomPoints(random_vector, int(n_randoms))
    # convert to numpy array
    RA = np.asarray([rand.RA() for rand in random_vector.iterator()])
    DEC = np.asarray([rand.DEC() for rand in random_vector.iterator()])
    randoms = pd.DataFrame({"RA": RA, "DEC": DEC})
    del stomp_map, random_vector
    return randoms


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
