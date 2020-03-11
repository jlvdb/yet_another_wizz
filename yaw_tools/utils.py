import argparse
import os
import subprocess
import sys
from shutil import rmtree

import numpy as np
from numpy import ma

from Nz_Fitting import format_variable, RedshiftData, RedshiftDataBinned

from .folders import (DEFAULT_EXT_BOOT, DEFAULT_EXT_COV, DEFAULT_EXT_DATA,
                      get_bin_key, ScaleFolder)


DEFAULT_CAT_EXT = 1
DEFAULT_COSMOLOGY = "default"
DEFAULT_RESAMPLING = 10
DEFAULT_PAIR_WEIGHTING = "Y"

DEFAULT_HDATA = "col 1 = mean redshift\n"
DEFAULT_HDATA += "col 2 = correlation amplitude ({:})\n"
DEFAULT_HDATA += "col 3 = correlation amplitude error"
DEFAULT_HBOOT = "correlation amplitude ({:}) realisations"
DEFAULT_HCOV = "correlation amplitude ({:}) covariance matrix"


def guess_bin_order(bin_keys):

    def get_zmin(key):
        zmin, zmax = [float(z) for z in key.split("z")]
        return zmin

    def get_zmax(key):
        zmin, zmax = [float(z) for z in key.split("z")]
        return zmax

    keys_sorted = sorted(bin_keys, key=get_zmin)
    # one bin could be the master sample
    zmin = np.inf
    zmax = 0.0
    for key in keys_sorted:
        zmin = min(zmin, get_zmin(key))
        zmax = max(zmax, get_zmax(key))
    master_key = None
    for i, key in enumerate(keys_sorted):
        if zmin == get_zmin(key) and zmax == get_zmax(key):
            master_key = keys_sorted.pop(i)
            break
    if master_key is not None:
        keys_sorted.append(master_key)
    return keys_sorted


def TypeNone(type):
    """Test input value for being 'type', but also accepts None as default

    Arguments:
        type [type]:
            specifies the type (int or float) the input argument has to obey
    Returns:
        type_text [function]:
            function that takes 'value' as argument and tests, if it is either
            None or of type 'type'
    """
    def type_test(value):
        if value is None:
            return value
        else:
            strtype = "float" if type == float else "int"
            try:
                return type(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "invalid %s value: '%s'" % (strtype, value))
    return type_test  # closure being of fixed type


def tex2png(texfile, pngfile=None, dpi=600, verbose=False):
    # format the latex template string
    latex_string = r"\documentclass[border=2pt]{standalone}" + "\n"
    latex_string += r"\usepackage{amsmath}" + "\n"
    latex_string += r"\usepackage[separate-uncertainty]{siunitx}" + "\n"
    latex_string += r"\begin{document}" + "\n"
    with open(texfile) as f:
        latex_string += "\n".join(f.readlines())
    latex_string += r"\end{document}" + "\n"

    if pngfile is None:
        pngfile = texfile
    pngfile = os.path.splitext(pngfile)[0]

    tmpdir = "tex2png_%s" % os.urandom(6).hex()
    os.mkdir(tmpdir)
    try:
        basename = os.path.splitext(os.path.basename(texfile))[0]
        basepath = os.path.join(tmpdir, basename)
        if verbose:
            print("reading input TEX file: %s" % texfile)
        with open(basepath + ".tex", "w") as f:
            f.write(latex_string)
        # run pdflatex
        with open(os.devnull, "w") as pipe:
            returncode = subprocess.call([
                "pdflatex", "-interaction=batchmode",
                "-jobname=%s" % basepath, basepath,],
                stdout=pipe, stderr=pipe)
        # report errors if exited unexpected
        if returncode:
            # display the first error reported in the log
            print("#" * 40)
            with open(basepath + ".log") as f:
                lines = f.readlines()
                for start, line in enumerate(lines):
                    if "!" in line:
                        end = start
                        while end < len(lines):
                            if "!" not in lines[end]:
                                break
                            end += 1
                        break
            for i in range(max(start - 5, 0), min(end + 5, len(lines))):
                print(lines[i].strip("\n"))
            print("#" * 40)
            sys.exit("ERROR:something went wrong during conversion to PDF")
        # run pdftocairo
        with open(basepath + ".log", "w") as pipe:
            returncode = subprocess.call([
                "pdftocairo", "-singlefile", "-png", "-r", str(dpi),
                basepath + ".pdf", pngfile], stdout=pipe, stderr=pipe)
        # report errors if exited unexpected
        if returncode:
            # display the log
            print("#" * 40)
            with open(basepath + ".log") as f:
                lines = f.readlines()
                for line in lines:
                    print(line.strip("\n"))
            print("#" * 40)
            sys.exit("ERROR:something went wrong during conversion to PNG")
        if verbose:
            print("created output PNG:     %s.png" % pngfile)
    except OSError:
        print(
            "WARNING: pdflatex or pdftocairo are not availble, could not "
            "convert TEX to PNG")
    finally:
        rmtree(tmpdir)


def write_fit_stats(fitparams, folder, precision=3, notation="decimal"):
    if not os.path.exists(folder):
        os.mkdir(folder)
    # write tex files
    with open(os.path.join(folder, "chi_squared.tex"), "w") as f:
        chisq = fitparams.chiSquare()
        string = format_variable(
            chisq, error=None, precision=precision, notation=notation)
        f.write("$\\chi^2 = %s$\n" % string.strip(" $"))
    with open(os.path.join(folder, "chi_squared_ndof.tex"), "w") as f:
        chisq /= fitparams.nDoF()
        string = format_variable(
            chisq, error=None, precision=precision, notation=notation)
        f.write("$\\chi^2_{\\rm dof} = %s$\n" % string.strip(" $"))


def write_parameters(
        fitparams, folder, precision=3, notation="auto", to_png=True):
    # construct the header
    name_header = "# name"
    header = " ".join(fitparams.names)
    maxwidth = max(len(name) for name in fitparams.names)
    maxwidth = max(len(name_header), maxwidth)
    # create the output directory
    if not os.path.exists(folder):
        os.mkdir(folder)
    # write tex files for each parameter
    for name in fitparams.names:
        texfile = os.path.join(folder, "%s.tex" % name)
        with open(texfile, "w") as f:
            f.write("%s\n" % fitparams.paramAsTEX(
                name, precision=precision, notation=notation))
        # convert to PNG
        if to_png:
            tex2png(texfile)
    # write a list of best fit parameters
    with open(os.path.join(folder, "parameters" + DEFAULT_EXT_DATA), "w") as f:
        f.write(
            "{:<{w}}    {:<16}    {:<16}\n".format(
                name_header, "value", "error", w=maxwidth))
        for name in fitparams.names:
            f.write(
                "{:>{w}}    {: 16.9e}    {: 16.9e}\n".format(
                    name, fitparams.paramBest(name),
                    fitparams.paramError(name), w=maxwidth))
    # write a list of bootstrap realisations
    np.savetxt(
        os.path.join(folder, "parameters" + DEFAULT_EXT_BOOT),
        fitparams.paramSamples(), header=header)
    # write the parameter covariance
    np.savetxt(
        os.path.join(folder, "parameters" + DEFAULT_EXT_COV),
        fitparams.paramCovar(), header=header)


def write_nz_stats(statdir, data, zkey=None):
    # write mean and median redshifts
    if not os.path.exists(statdir):
        os.mkdir(statdir)
    # write tex files
    iterator = zip(
        ("mean", "median"), ("\\langle z \\rangle", "z_{\\rm med}"))
    for stat, TEX in iterator:
        if zkey is None:
            statfile = os.path.join(statdir, "%s.tex" % stat)
        else:
            statfile = os.path.join(statdir, "%s_%s.tex" % (stat, zkey))
        try:
            val, err = getattr(data, stat)(error=True)
        except TypeError:  # RedshiftHistogram does not support errors
            val = getattr(data, stat)()
            err = None
        with open(statfile, "w") as f:
            string = format_variable(
                val, error=err, TEX=True, precision=3, notation="decimal",
                use_siunitx=True)
            f.write("$%s = %s$\n" % (TEX, string.strip("$")))
        # convert to PNG
        tex2png(statfile)


def write_nz_data(
        path, data, hdata=None, hboot=None, hcov=None, stats=False,
        dtype_message="n(z)"):
    assert(type(data) is RedshiftData)
    basepath = os.path.splitext(path)[0]
    print(
        "writing {:} data to: {:}.*".format(
            dtype_message, os.path.basename(basepath)))
    data.write(basepath, hdata, hboot, hcov)
    if stats:
        statdir = os.path.join(os.path.dirname(path), "stats")
        try:
            zkey = get_bin_key(basepath)
        except ValueError:
            zkey = None
        write_nz_stats(statdir, data, zkey=zkey)


def write_global_cov(folder, data, order, header, prefix):
    assert(isinstance(folder, ScaleFolder))
    assert(type(data) is RedshiftDataBinned)
    print("writing global covariance matrix to: %s" % folder.basename())
    path = folder.path_global_cov_file(prefix)
    data.writeCovMat(os.path.splitext(path)[0], head=header)
    # store the order for later use
    with open(folder.path_bin_order_file(), "w") as f:
        for zbin in order:
            f.write("%s\n" % zbin)
