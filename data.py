import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz


class CCdata(object):
    """
    Container for loading and storing clustering redshift data from a file.
      Parameters:
        filepath [string]: ascii file which contains a Nx3-shaped data table
        zmax [float]: mask out datapoints with redshifts above this value
    """

    def __init__(self, z, y, dy, zmax=None, remove_nans=True):
        assert(len(z) == len(y))
        assert(len(z) == len(dy))
        data = np.stack([z, y, dy]).T
        # mask the values if required
        if zmax is not None:
            data = data[data[:, 0] <= zmax]
        self._data = data.T
        # remove nans
        if remove_nans:
            self._data_mask = ~np.isnan(self._data.T).any(axis=1)
            if np.any(self._data[2] > 0.0):
                self._data_mask = np.logical_and(
                    self._data_mask,
                    self._data[2] != 0.0)
            self._data = np.transpose(
                self._data.T[self._data_mask])
        else:
            self._data_mask = np.ones(self.data.shape[1], dtype="bool")
        self._npoints = self._data.shape[1]

    def __len__(self):
        return self._npoints

    def __copy__(self):
        return self.__class__(self.filename)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        else:
            return np.all(self.getData() == other.getData())

    def __getattr__(self, key):
        for idx, name in zip(range(3), ["z", "y", "dy"]):
            if name == key:
                return self._data[idx]
        if key == "pdf":
            masked_z = self.z[self._data_maskmask]
            masked_y = self.y[self._data_maskmask]
            return masked_y / np.trapz(masked_y, x=masked_z)
        elif key == "cdf":
            masked_z = self.z[self._data_maskmask]
            masked_y = self.y[self._data_maskmask]
            return cumtrapz(self.pdf, x=masked_z, initial=0.0)
        else:
            raise AttributeError("no attribute named '%s'" % key)

    def normalisation(self, zmax=2.5):
        """
        Normalise a redshift distribution by trapezodial intergration.
        Negative data points are removed by averaging neighbouring data points.
          Parameters:
            zmax [float]: upper redshift limit for normalisation
          Returns:
            norm [float]: normalisation constant
        """
        # make sure the original values are not modified
        z, y, dy = self.getData(normed=False)
        # mask out high redshift tail
        zmask = z <= zmax
        idx_max = len(self) + 1
        idx_limit = sum(zmask)  # index limit == sum of unmasked values
        for I in range(idx_limit):
            # average neighbouring data points if the current one is negative
            if y[I] < 0.0:
                # step outwards in both directions until hitting boundaries
                for i in range(1, I):
                    # the interval is n = [I - i, ..., I, ...,  I + i]
                    imin = max(I - i, 0)
                    imax = min(I + i, idx_max)
                    y_select = y[imin:imax]
                    dy_select = dy[imin:imax]
                    # compute the average weighted by the inverse variance
                    var_inverse_sum = np.sum(dy_select ** -2)
                    y_new = np.sum(y_select / dy_select**2) / var_inverse_sum
                    dy_new = np.sqrt(1 / var_inverse_sum)
                    # stop expanding interval, if new value is positive
                    if y_new >= 0.0:
                        # assign average values to data points within interval
                        y[imin:imax] = y_new
                        dy[imin:imax] = dy_new
                        break
        # compute the normalisation with trapezodial rule integration
        norm = np.trapz(y[zmask], x=z[zmask])
        return norm

    def mean(self):
        return np.average(self.z[self._mask], weights=self.pdf)

    def median(self):
        return np.interp(0.5, self.cdf, self.z[self._mask])

    def getData(self, normed=False, zmax=2.5):
        """
        Obtain a copy of the current data.
          Parameters:
            normed [bool]: wether the data should be normalised
            zmax [float]: upper redshift limit for normalisation
          Returns:
            data [array, 3xN]: array with layout (x, y, dy)
        """
        if normed:
            norm = self.normalisation(zmax)
            norm_vec = np.asarray([1.0, norm, norm])
            data = self._data.copy() / norm_vec[:, np.newaxis]
        else:
            data = self._data.copy()
        if zmax is not None:
            data = data[:, self.z <= zmax]
        return data

    def getMask(self):
        """
        Obtain a copy of the mask for the current data
          Returns:
            data [array]: bool array matching original data length
        """
        return self._data_mask.copy()

    def plot(self, normed=False, zmax=None, axis=None, rescale=1.0, **kwargs):
        """
        Make an errorbar plot of the data.
          Parameters:
            normed [bool]: wether the data should be normalised
            zmax [float]: upper redshift limit for plot
            axis [matplotlib.axis]: axis instance to plot at
            kwargs: optional parameters for pyplot.errorbar
          Returns:
            color [string]: matplotlib color string of the plotted data group
        """
        # set default plot layout
        if "color" not in kwargs:
            kwargs["color"] = "k"
        if "linestyle" in kwargs:  # only use 'ls' alias
            kwargs["ls"] = kwargs.pop("linestyle")
        if "ls" not in kwargs:
            kwargs["ls"] = "none"
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        # get axis object
        if axis is None:
            axis = plt.gca()
        # apply plot limits
        if zmax is None:
            z, y, dy = self.getData(normed)
            axis.set_xlim(xmin=0.0)
        else:
            # find the index + 1 belonging to xmax
            zmask = self.z <= zmax
            # index limit == sum of unmasked values, or array length
            idx_limit = min(sum(zmask), len(self))
            if kwargs["ls"] != "none":
                idx_limit = min(idx_limit + 1, len(self))
            # mask the data to the x-limit
            z, y, dy = self.getData(normed)[:, :idx_limit]
            axis.set_xlim([0.0, zmax])
        # make errorbar plot
        line, _, _ = axis.errorbar(z, y * rescale, dy * rescale, **kwargs)
        color = line.get_color()
        return color


class zhistogram(object):

    def __init__(self, zdata, binning, weights=None, bin_mask=None):
        if bin_mask is not None:
            self.data = zdata[bin_mask]
            if weights is not None:
                self.weights = weights[bin_mask]
            else:
                self.weights = np.ones(np.count_nonzero(bin_mask))
        else:
            self.data = zdata
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.ones(np.count_nonzero(bin_mask))
        self.weights /= self.weights.mean()
        self.set_binning(binning)

    def total_weight(self):
        return self.weights.sum()

    def weights_on(self):
        self._rebin(True)

    def weights_off(self):
        self._rebin(False)

    def _rebin(self, use_weight=True):
        bin_idx = np.digitize(self.data, bins=self.binning) - 1
        # objects in each bin
        z = np.empty(len(self.binning) - 1)
        counts = np.empty_like(z)
        for i in range(len(z)):
            mask = bin_idx == i
            if np.count_nonzero(mask) == 0:
                z[i] = np.nan
                counts[i] = 0.0
            else:
                if self.weights is not None and use_weight:
                    weight = self.weights[mask]
                else:
                    weight = np.ones(np.count_nonzero(mask))
                data = self.data[mask]
                z[i] = np.average(data, weights=weight)
                counts[i] = np.sum(weight)
        counts /= np.diff(self.binning)
        mask = np.isfinite(z)
        self.z = z[mask]
        self.pdf = counts[mask] / np.trapz(counts[mask], x=self.z)
        self.cdf = cumtrapz(self.pdf, x=self.z, initial=0.0)

    def set_binning(self, binning):
        assert(np.all(np.diff(binning) > 0.0))
        self.binning = binning
        self._rebin()

    def mean(self):
        return np.average(self.z, weights=self.pdf)

    def median(self):
        return np.interp(0.5, self.cdf, self.z)
