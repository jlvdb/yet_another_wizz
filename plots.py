from difflib import get_close_matches

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from yaw_tools.data import CCdata


def subplot_grid(
        n_plots, n_cols, sharex=True, sharey=True, scale=1.0, dpi=None):
    n_cols = min(n_cols, n_plots)
    n_rows = (n_plots // n_cols) + min(1, n_plots % n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, sharex=sharex, sharey=sharey,
        dpi=dpi, figsize=(
            scale * (0.5 + 3.8 * n_cols),
            scale * (0.5 + 2.8 * n_rows)))
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.T
    for i in range(n_cols * n_rows):
        i_row, i_col = i // n_cols, i % n_cols
        if i < n_plots:
            axes[i_row, i_col].tick_params(
                "both", direction="in",
                bottom=True, top=True, left=True, right=True)
        else:
            fig.delaxes(axes[i_row, i_col])
    fig.set_facecolor("white")
    return fig, axes


def hist_step_filled(
        data, bins, color, label=None, ax=None, normed=True, rescale=1.0,
        weight=None):
    count = np.histogram(
        data, bins, density=normed, weights=weight)[0] * rescale
    y = np.append(count[0], count)
    if ax is None:
        ax = plt.gca()
    ax.fill_between(
        bins, y, 0.0, color=color, step="pre", alpha=0.4, label=label)
    ax.step(bins, y, color=color)


def hist_smooth_filled(
        data, bins, color, label=None, ax=None, normed=True, kwidth="scott"):
    z = np.linspace(bins[0], bins[-1], 251)
    kde = gaussian_kde(data, bw_method=kwidth)
    y = kde(z)
    if not normed:
        phot_y *= len(photz)
    if ax is None:
        ax = plt.gca()
    ax.fill_between(z, phot_y, 0.0, color=color, alpha=0.4, label=label)
    ax.plot(z, phot_y, color=color)


def scatter_dense(xdata, ydata, color, s=1, ax=None, alpha=0.25):
    if ax is None:
        ax = plt.gca()
    handle = ax.scatter(
        xdata, ydata, s=s, color=color, marker=".", alpha=alpha)
    hcolor = handle.get_facecolors()[0].copy()
    hcolor[-1] = 1.0
    handle = plt.scatter([], [], color=hcolor)
    return handle


class CCfigure(object):

    fig = None
    axes = None
    _bin_limits = None
    _hist = None
    _points = None
    _pdfs = None
    _handles = None
    _labels = None

    def __init__(self, n_plots, n_columns=3, dpi=200):
        self._n_plots = n_plots
        self._n_columns = n_columns
        self._dpi = dpi
        self.replot()

    def replot(self):
        self._handles = []
        self._labels = []
        # create an empty figure
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axes = subplot_grid(
            self._n_plots, self._n_columns,
            sharex=True, sharey=True, dpi=self._dpi)
        for ax in self.axes.flatten():
            ax.axhline(y=0.0, color="k", lw=0.5, zorder=-5)
            ax.grid(alpha=0.25)
        # indicate bin limits
        if self._bin_limits is not None:
            for ax, zlims in zip(self.axes.flatten(), self._bin_limits):
                ax.annotate(
                    r"$%.1f \leq z_{\rm phot} < %.1f$" % zlims, (0.95, 0.95),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=12)
        # add histogram
        if self._hist is not None:
            label = tuple(self._hist.keys())[0]
            for ax, zhist in zip(self.axes.flatten(), self._hist[label]):
                plot_counts = np.append(0.0, zhist.pdf)
                plot_counts = np.append(plot_counts, 0.0)
                plot_z = np.append(0.0, zhist.z)
                plot_z = np.append(plot_z, zhist.binning[-1])
                handle = ax.fill_between(
                    plot_z, plot_counts, 0.0, color="k", alpha=0.2,
                    label=label, zorder=-2)
            self._handles.append(handle)
            self._labels.append(label)
            self.ylabel(r"$n(z)$")
        # add data points
        if self._points is not None:
            for label, data in self._points.items():
                # try to normalize to any reference data
                if self._pdfs is not None:
                    pdf_key = get_close_matches(
                        label, self._pdfs.keys(), n=1, cutoff=0.0)[0]
                    pdf_resample = np.concatenate([
                        np.interp(d.z, pdf.z, pdf.y)
                        for d, pdf in zip(data, self._pdfs[pdf_key])])
                    # fit the amplitudes weighted by the error
                    norm = curve_fit(
                        lambda xdata, *params: pdf_resample * params[0],
                        np.concatenate([d.z for d in data]),
                        np.concatenate([d.y for d in data]), p0=1.0,
                        sigma=np.concatenate([d.dy for d in data]))[0][0]
                elif self._hist is not None:
                    hist_key = tuple(self._hist.keys())[0]
                    hist_concat = np.concatenate(
                        zhist.pdf for zhist in self._hist[hist_key])
                    # fit the amplitudes
                    norm = curve_fit(
                        lambda xdata, *params: hist_concat * params[0],
                        np.concatenate([d.z for d in data]),
                        np.concatenate([d.y for d in data]), p0=1.0)[0][0]
                else:
                    norm = 1.0
                # make the plot
                for ax, cc in zip(self.axes.flatten(), data):
                    handle = ax.errorbar(
                        cc.z, cc.y / norm, cc.dy / norm,
                        label=label, marker=".", ls="none")
                self._handles.append(handle)
                self._labels.append(label)
        # add pdfs
        if self._pdfs is not None:
            for label, data in self._pdfs.items():
                for ax, pdf in zip(self.axes.flatten(), data):
                    handle = ax.plot(
                        pdf.z, pdf.y, label=label)[0]
                self._handles.append(handle)
                self._labels.append(label)
            self.ylabel(r"$p(z)$")
        # finalize the plot
        self.xlim()
        self.xlabel(r"$z$")
        self.fig.tight_layout(h_pad=0.0, w_pad=0.0)
        self.fig.subplots_adjust(top=0.92, hspace=0.0, wspace=0.0)

    def close(self):
        self.fig.close()

    def add_bin_limits(self, zbin_limits):
        if len(zbin_limits) != len(self.axes.flatten()):
            raise ValueError("bin limits do not match number of bins")
        self._bin_limits = zbin_limits
        self.replot()

    def add_hist(self, datadict, keyorder, use_weights=True, label=None):
        if label is None:
            label = r"$z_{\rm true}$"
        self._hist = {label: []}
        for key in keyorder:
            zhist = datadict[key]
            if use_weights:
                zhist.weights_on()
            else:
                zhist.weights_off()
            self._hist[label].append(zhist)
        self.replot()

    def remove_hist(self):
        self._hist = None
        self.replot()

    def add_points(self, datadict, keyorder, label):
        if self._points is None:
            self._points = {}
        self._points[label] = []
        for key in keyorder:
            self._points[label].append(datadict[key])
        self.replot()

    def remove_points(self, label):
        self._points.pop(label)
        self.replot()

    def add_pdf(self, datadict, keyorder, label):
        if self._pdfs is None:
            self._pdfs = {}
        self._pdfs[label] = []
        for key in keyorder:
            data = datadict[key]
            norm = np.trapz(data.y, x=data.z)
            self._pdfs[label].append(
                CCdata(data.z, data.y / norm, data.dy / norm))
        self.replot()

    def remove_pdf(self, label):
        self._pdfs.pop(label)
        self.replot()

    def legend(self, size=None):
        self.fig.legend(
            handles=self._handles, labels=self._labels, loc="upper center",
            ncol=len(self._handles), frameon=False, prop={"size": 14})

    def xlim(self, left=0.0, right=None):
        if right is None:
            if self._points is not None:
                right = 0.0
                for label, data in self._points.items():
                    right = max(
                        right, np.round(data[0].z.max() + 0.1, decimals=1))
            elif self._hist is not None:
                right = 0.0
                for label, zhist in self._points.items():
                    right = max(
                        right, np.round(zhist.binning[-1] + 0.1, decimals=1))
            elif self._pdfs is not None:
                right = 0.0
                for label, data in self._pdfs.items():
                    right = max(
                        right, np.round(data[0].z.max() + 0.1, decimals=1))
        for ax in self.axes.flatten():
            ax.set_xlim(left, right)

    def ylim(self, bottom=None, top=None):
        for ax in self.axes.flatten():
            ax.set_ylim(bottom, top)

    def xlabel(self, label, fontsize=12):
        for i, ax in enumerate(self.axes.flatten()):
            if i // self._n_columns == (self._n_plots - 1) // self._n_columns:
                ax.set_xlabel(label, fontsize=fontsize)

    def ylabel(self, label, fontsize=12):
        for i, ax in enumerate(self.axes.flatten()):
            if i % self._n_columns == 0:
                ax.set_ylabel(label, fontsize=fontsize)

    def title(self, title, fontsize=16):
        self.fig.suptitle(title, fontsize=fontsize)


class CCtest(object):

    _ntype_list = ("z_hist", "w_sp", "n_tilde", "n_CC", "n_fit")
    _ntype_label = {
        "z_hist": r"$z_{\rm true}$", "w_sp": r"$w_{\rm sp}$",
        "n_tilde": r"$\tilde n$", "n_CC": r"$n_{\rm CC}$",
        "n_fit": r"$n_{\rm fit}$"}
    default_order = (
        "0.101z0.301", "0.301z0.501", "0.501z0.701",
        "0.701z0.901", "0.901z1.201", "0.101z1.201")
    zbin_limits = (
        (0.101, 0.301), (0.301, 0.501), (0.501, 0.701),
        (0.701, 0.901), (0.901, 1.201), (0.101, 1.201))
    w_sp = None
    w_ss = None
    w_pp = None
    z_hist = None
    _fittype = None
    alpha = None
    alpha_err = None
    chi_ndof = None

    def __init__(self, w_sp_glob, name, weighted=False):
        self.weighted = weighted
        self.name = name
        self.binning, self.w_sp = load_cc_sample(w_sp_glob)

    def _bias_model(self, z, *param):
        return param[0] * (z + 1.0) ** param[1]

    def _compute_alpha(self, ntype="n_tilde"):
        assert(ntype in self._ntype_list)

        def chisquares(args, bin_pdfs, full_pdf, weights):
            # compute the weighted sum of bin counts
            bin_ys = []
            for i, pdf in enumerate(bin_pdfs):
                bias = self._bias_model(pdf.z, 1.0, args[0])  # args = [alpha]
                y = pdf.y / bias
                norm = np.trapz(y, x=pdf.z)
                bin_ys.append(y / norm * weights[i])
            stacked_ys = np.sum(bin_ys, axis=0)
            # compute the full sample
            bias = self._bias_model(full_pdf.z, 1.0, args[0])
            full_y = full_pdf.y / bias
            norm = np.trapz(full_y, x=full_pdf.z)
            full_y /= norm
            full_err = full_pdf.dy / bias / norm
            # compute the chisquared between stacked and full pdf
            chisq = ((stacked_ys - full_y) / full_err)**2
            return chisq.sum()

        def lnprob(args, bin_pdfs, full_pdf, weights):
            chisq = chisquares(args, bin_pdfs, full_pdf, weights)
            return -0.5 * chisq

        if ntype != self._fittype:
            weights_dict = self.get_bin_weights()
            weights = np.asarray([
                weights_dict[key] for key in self.default_order[:-1]])
            n_tilde = getattr(self, "get_" + ntype)()
            nofz = [n_tilde[key] for key in self.default_order]

            ndim = 1
            nwalkers = 20 * ndim
            nsteps = 2000
            ndof = len(nofz[-1]) - ndim

            args = (nofz[:-1], nofz[-1], weights)
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, args=args)
            p0 = np.random.normal(
                np.full(ndim, 0.0), np.full(ndim, 0.25),
                size=(nwalkers, ndim))
            sampler.run_mcmc(p0, nsteps)
            names = ["alpha"]
            labels = ["\\alpha"]
            samples = MCSamples(
                samples=sampler.chain[:, nsteps // 10:, :].reshape((-1, ndim)),
                loglikes=sampler.lnprobability[:, nsteps // 10:].flatten(),
                names=names, labels=labels)
            samples.removeBurn(remove=0.2)
            self._fittype = ntype
            self.alpha = samples.mean(names)
            self.alpha_err = samples.std(names)
            self.chi_ndof = chisquares(self.alpha, *args) / ndof
        print("best fit (%s): %.3f +- %.3f (chi^2 / ndof = %.3f)" % (
            self.name, self.alpha, self.alpha_err, self.chi_ndof))

    def add_z_hist(self, fpath, zspec_key, zphot_key, weight_key=None):
        if self.weighted:
            assert(weight_key is not None)
        self.z_hist = {}
        with pyfits.open(fpath) as fits:
            z_spec = fits[1].data[zspec_key]
            z_b = fits[1].data[zphot_key]
            if weight_key is not None:
                weight = fits[1].data["recal_weight"]
            else:
                weight = None
            for i, (zlim, key) in enumerate(
                    zip(self.zbin_limits, self.default_order)):
                mask = (z_b >= zlim[0]) & (z_b < zlim[1])
                self.z_hist[key] = zhistogram(
                    z_spec, self.binning, weights=weight, bin_mask=mask)
                if self.weighted:
                    self.z_hist[key].weights_on()
                else:
                    self.z_hist[key].weights_off()

    def add_w_ss(self, w_ss_path):
        data = np.loadtxt(w_ss_path)
        try:
            z, cc, err = data[:, [0, 1, 3]].T
        except IndexError:
            z, cc, err = data[:, [0, 1, 2]].T
        inf_mask = np.abs(cc) == np.inf
        cc[inf_mask] = np.nan
        err[inf_mask] = np.inf
        self.w_ss = CCdata(z, cc, err)

    def add_w_pp(self, w_pp_path):
        data = np.loadtxt(w_pp_path)
        try:
            z, cc, err = data[:, [0, 1, 3]].T
        except IndexError:
            z, cc, err = data[:, [0, 1, 2]].T
        inf_mask = np.abs(cc) == np.inf
        cc[inf_mask] = np.nan
        err[inf_mask] = np.inf
        self.w_pp = CCdata(z, cc, err)

    def get_z_hist(self):
        if self.z_hist is not None:
            return self.z_hist
        else:
            raise AssertionError("redshift histogram data required")

    def get_w_sp(self):
        if self.w_sp is not None:
            return self.w_sp
        else:
            raise AssertionError("cross-correlation amplitudes required")

    def get_w_ss(self):
        if self.w_ss is not None:
            return self.w_ss
        else:
            raise AssertionError(
                "reference auto-correlation amplitudes required")

    def get_w_pp(self):
        if self.w_pp is not None:
            return self.w_pp
        else:
            raise AssertionError(
                "unknown auto-correlation amplitudes required")

    def get_stat(self, ntype, errors=False):
        data_dict = getattr(self, "get_" + ntype)()
        stat_mean = {key: data.mean() for key, data in data_dict.items()}
        stat_median = {key: data.median() for key, data in data_dict.items()}
        stats = {"mean": stat_mean, "median": stat_median}
        # compute errors from random realisations
        if errors:
            stats["mean_err"] = {}
            stats["median_err"] = {}
            for key, data in data_dict.items():
                means = np.empty(1000)
                medians = np.empty_like(means)
                for i in range(len(means)):
                    realisation = CCdata(
                        data.z, np.random.normal(data.y, data.dy), data.dy)
                    means[i] = realisation.mean()
                    medians[i] = realisation.median()
                stats["mean_err"][key] = means.std()
                stats["median_err"][key] = medians.std()
        return stats

    def get_bin_weights(self):
        weights = {
            key: hist.total_weight()
            for key, hist in self.get_z_hist().items()}
        total_weight = sum(weights.values()) / 2  # since full sample included
        normed_weights = {
            key: weight / total_weight for key, weight in weights.items()}
        return normed_weights

    def get_bias_term(self):
        w_sspp = np.sqrt(self.get_w_ss().y * self.get_w_pp().y)
        w_err = np.sqrt(
            0.5 * self.get_w_pp().y / self.get_w_ss().y *
            self.get_w_ss().dy**2 +
            0.5 * self.get_w_ss().y / self.get_w_pp().y *
            self.get_w_pp().dy**2)
        return CCdata(self.get_w_ss().z, w_sspp, w_err)

    def get_bias_fit(self, ntype="n_tilde"):
        self._compute_alpha(ntype)
        z = self.get_w_pp().z
        B_alpha = self._bias_model(z, 1.0, self.alpha)
        B_error = np.abs(B_alpha * np.log(z + 1.0) * self.alpha_err)
        # normalize to median of sqrt(w_pp)
        norm = np.median(B_alpha) / np.nanmedian(np.sqrt(self.get_w_pp().y))
        B_alpha /= norm
        B_error /= norm
        return CCdata(z, B_alpha, B_error)

    def get_n_tilde(self):
        n_tilde = {
            key: w_sp.y / np.sqrt(self.get_w_ss().y)
            for key, w_sp in self.get_w_sp().items()}
        n_error = {
            key: np.sqrt(
                w_sp.dy**2 / self.get_w_ss().y +
                n_tilde[key] * (self.get_w_ss().dy / self.get_w_ss().y)**2)
            for key, w_sp in self.get_w_sp().items()}
        data = {}
        for key in self.default_order:
            data[key] = CCdata(
                self.get_w_sp()[key].z, n_tilde[key], n_error[key])
        return data

    def get_n_CC(self):
        n_CC = {
            key: w_sp.y / np.sqrt(self.get_w_ss().y * self.get_w_pp().y)
            for key, w_sp in self.get_w_sp().items()}
        n_err = {
            key: np.sqrt(
                w_sp.dy**2 / (self.get_w_ss().y * self.get_w_pp().y) +
                n_CC[key] * (self.get_w_ss().dy / self.get_w_ss().y)**2 +
                n_CC[key] * (self.get_w_pp().dy / self.get_w_pp().y)**2)
            for key, w_sp in self.get_w_sp().items()}
        data = {}
        for key in self.default_order:
            data[key] = CCdata(self.get_w_sp()[key].z, n_CC[key], n_err[key])
        return data

    def get_n_fit(self, ntype="n_tilde"):
        bias = self.get_bias_fit(ntype)
        n_tilde = self.get_n_tilde()
        n_fit = {key: n_tilde[key].y / bias.y for key in self.default_order}
        n_err = {
            key: np.sqrt(
                (n_tilde[key].dy / bias.y)**2 +
                (n_fit[key] * bias.dy / bias.y)**2)
            for key in self.default_order}
        data = {}
        for key in self.default_order:
            data[key] = CCdata(n_tilde[key].z, n_fit[key], n_err[key])
        return data

    def bias_figure(self, others=None):
        if others is None:
            others = []
        elif isinstance(others, self.__class__):
            others = [others]
        for other in others:
            assert(isinstance(other, self.__class__))
        others.insert(0, self)

        fig, axes = subplot_grid(3, 3, sharex=True, sharey=False, dpi=200)
        for c, other in enumerate(others):
            # plot w_ss
            w = other.get_w_ss()
            outl_mask = (
                (np.abs(w.y) < np.nanmean(w.y) + 3 * np.nanstd(w.y)) &
                np.isfinite(w.y) & np.isfinite(w.dy))
            axes[0, 0].errorbar(
                w.z[outl_mask], w.y[outl_mask], yerr=w.dy[outl_mask],
                marker=".", ls="none", color="C%d" % c,
                label=r"$w_{\rm ss}$ (%s)" % other.name)
            axes[0, 0].legend()
            # plot w_pp
            try:
                w = other.get_w_pp()
                outl_mask = (
                    (np.abs(w.y) < np.nanmean(w.y) + 3 * np.nanstd(w.y)) &
                    np.isfinite(w.y) & np.isfinite(w.dy))
                axes[0, 1].errorbar(
                    w.z[outl_mask], w.y[outl_mask], yerr=w.dy[outl_mask],
                    marker=".", ls="none", color="C%d" % c,
                    label=r"$w_{\rm pp}$ (%s)" % other.name)
                bias = other.get_bias_fit("n_tilde")
                bias_sq = CCdata(bias.z, bias.y**2, 2 * bias.y * bias.dy)
                axes[0, 1].plot(
                    bias_sq.z, bias_sq.y, color="C%d" % c,
                    label=r"$\mathcal{B}_\alpha^2$ (%s)" % other.name)
                axes[0, 1].fill_between(
                    bias_sq.z, bias_sq.y - bias_sq.dy,
                    bias_sq.y + bias_sq.dy,
                    color="C%d" % c, alpha=0.2)
                axes[0, 1].legend()
            except AssertionError:
                axes[0, 1].annotate(
                    "n/a", (0.5, 0.5), xycoords="axes fraction",
                    va="center", ha="center", size=25, color="0.5")
            # plot bias correction terms
            # plot sqrt(w_ss w_pp)
            try:
                w = other.get_bias_term()
                outl_mask = (
                    (np.abs(w.y) < np.nanmean(w.y) + 3 * np.nanstd(w.y)) &
                    np.isfinite(w.y) & np.isfinite(w.dy))
                axes[0, 2].errorbar(
                    w.z[outl_mask], w.y[outl_mask], yerr=w.dy[outl_mask],
                    marker=".", ls="none", color="C%d" % c,
                    label=r"$\sqrt{w_{\rm ss} w_{\rm pp}}$ (%s)" % other.name)
                # plot sqrt(w_ss) B_alpha
                val = np.sqrt(other.get_w_ss().y) * bias.y
                err = np.sqrt(
                    (np.sqrt(other.get_w_ss().y) * bias.dy)**2 +
                    (bias.y * 0.5 / np.sqrt(other.get_w_ss().y) *
                     other.get_w_ss().dy)**2)
                outl_mask = np.abs(val) < np.nanmean(val) + 3 * np.nanstd(val)
                w = CCdata(w.z[outl_mask], val[outl_mask], err[outl_mask])
                axes[0, 2].plot(
                    w.z, w.y, color="C%d" % c,
                    label=r"$\sqrt{w_{\rm ss}} \mathcal{B}_\alpha$ (%s)" %
                          other.name)
                axes[0, 2].fill_between(
                    w.z, w.y - w.dy, w.y + w.dy,
                    color="C%d" % c, alpha=0.2)
                axes[0, 2].legend()
            except AssertionError:
                axes[0, 2].annotate(
                    "n/a", (0.5, 0.5), xycoords="axes fraction",
                    va="center", ha="center", size=25, color="0.5")

        for ax in axes[0]:
            ax.grid(alpha=0.25)
            ax.set_xlim(0.0)
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0.0, ymax * 1.5)
            ax.set_xlabel("redshift")

        return fig

    def n_figure(self, ntypes, others=None):
        if type(ntypes) is str:
            ntypes = [ntypes]
        for ntype in ntypes:
            assert(ntype in self._ntype_list)

        if others is None:
            others = []
        elif isinstance(others, self.__class__):
            others = [others]
        for other in others:
            assert(isinstance(other, self.__class__))
        others.insert(0, self)

        fig = CCfigure(6, 3, self.zbin_limits)
        if self.z_hist is not None:
            fig.add_true(
                self.get_z_hist(), self.default_order,
                use_weights=self.weighted, label=self._ntype_label["z_hist"])
        for other in others:
            for ntype in ntypes:
                fig.add_cclayer(
                    getattr(other, "get_" + ntype)(), other.default_order,
                    label=self._ntype_label[ntype] + " (%s)" % other.name)
        fig.legend()
        fig.xlim(0.0)
        fig.xlabel("redshift")
        return fig

    def stat_figure(self, ntypes, others=None):
        if type(ntypes) is str:
            ntypes = [ntypes]
        for ntype in ntypes:
            assert(ntype in self._ntype_list)

        if others is None:
            others = []
        elif isinstance(others, self.__class__):
            others = [others]
        for other in others:
            assert(isinstance(other, self.__class__))
        others.insert(0, self)

        fig, [[ax1, ax2]] = subplot_grid(
            2, 2, sharex=False, sharey=True, dpi=200)
        ys = list(range(0, -len(self.default_order), -1))
        n_samples = len(others) * len(ntypes)

        handles = []
        labels = []
        xlim = 0.0
        for stat_key, ax in zip(["mean", "median"], [ax1, ax2]):
            ax.axvline(x=0.0, color="k")
            n = 0
            for other, marker in zip(others, "osDPv^<>"):
                ztrue_dict = other.get_stat("z_hist")[stat_key]
                for c, ntype in enumerate(ntypes):
                    stat_dict = other.get_stat(ntype, errors=True)
                    val_dict = stat_dict[stat_key]
                    err_dict = stat_dict[stat_key + "_err"]
                    handle = ax.errorbar(
                        [val_dict[key] - ztrue_dict[key]
                         for key in self.default_order],
                        ys + np.linspace(-0.1, 0.1, n_samples)[n],
                        xerr=[err_dict[key] for key in self.default_order],
                        marker=".", markersize=10, ls="none")
                    n += 1
                    if stat_key == "mean":
                        handles.append(handle)
                        labels.append(
                            self._ntype_label[ntype] + " (%s)" % other.name)
            ax.set_yticks(ys)
            ax.set_yticklabels([
                r"$%.1f \leq z_{\rm phot} < %.1f$" % zlims
                for zlims in self.zbin_limits])
            ax.set_xlabel(r"deviation in %s $z$" % stat_key)
            ax.grid(alpha=0.25)
            xlim = max(xlim, max(np.abs(ax.get_xlim())))

        for ax in [ax1, ax2]:
            ax.set_xlim(-xlim, xlim)
        fig.legend(
            handles=handles, labels=labels, loc="upper center",
            ncol=n_samples, frameon=False)
        fig.tight_layout(h_pad=0.0, w_pad=0.0)
        fig.subplots_adjust(top=0.9, hspace=0.0, wspace=0.05)
        return fig

    def stat_table(self, ntypes, others=None, stat_key="mean", nice=False):
        assert(stat_key in ("mean", "median"))
        if type(ntypes) is str:
            ntypes = [ntypes]
        for ntype in ntypes:
            assert(ntype in self._ntype_list)

        if others is None:
            others = []
        elif isinstance(others, self.__class__):
            others = [others]
        for other in others:
            assert(isinstance(other, self.__class__))
        others.insert(0, self)

        tab = Table()
        # this sadly ends in line wrapping
        if nice:
            tab.add_column(Column([
                r"$%s \leq z_{\rm phot} < %s$" % tuple(
                    k[:3] for k in key.split("z"))
                for key in self.default_order], "Bin"))
        else:
            tab.add_column(Column(self.default_order, "Bin"))
        # add truth
        ztrue_dict = self.get_stat("z_hist")[stat_key]
        tab.add_column(Column(
            [ztrue_dict[key] for key in self.default_order],
            "true %s-z" % stat_key, format="%.3f"))
        # add requested n(z) types
        for other in others:
            for c, ntype in enumerate(ntypes):
                stat_dict = other.get_stat(ntype)[stat_key]
                tab.add_column(Column(
                    [stat_dict[key] for key in self.default_order],
                    "%s %s-z (%s)" % (ntype, stat_key, other.name),
                    format="%.3f"))
        return tab
