import json
import os

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits

from .folders import get_bin_key


def get_bin_weights(framelist, filelist):
    weight_dict = {}
    for frame, path in zip(framelist, filelist):
        try:
            weight = frame.weights.sum()
        except AttributeError:  # assume uniform weights
            weight = len(frame)
        try:
            key = get_bin_key(path)
        except ValueError:
            key = "all"
        weight_dict[key] = weight
    return weight_dict


def bin_table(
        bindir, filepath, ra_name, dec_name, z_name, weight_name=None,
        region_name=None, zbins=None, cat_ext=1):
    # read input catalogue
    with pyfits.open(filepath) as fits:
        head = fits[cat_ext].header
        data = fits[cat_ext].data
    print("loaded %d objects" % len(data))
    # make catalogue for each selected bin
    framelist = []
    filelist = []
    if zbins is None:
        if z_name is None:
            filename = bindir.join("bin_all.fits")
            os.symlink(filepath, filename)
            frame = pd.DataFrame({
                "ra": data[ra_name].byteswap().newbyteorder(),
                "dec": data[dec_name].byteswap().newbyteorder()})
        else:
            zmin, zmax = data[z_name].min(), data[z_name].max()
            filename = bindir.zbin_filename(zmin, zmax, ".fits", prefix="bin")
            os.symlink(filepath, filename)
            frame = pd.DataFrame({
                "ra": data[ra_name].byteswap().newbyteorder(),
                "dec": data[dec_name].byteswap().newbyteorder(),
                "z": data[z_name].byteswap().newbyteorder()})
        if weight_name is not None:
            frame["weights"] = data[weight_name].byteswap().newbyteorder()
        if region_name is not None:
            frame["region_idx"] = data[region_name].byteswap().newbyteorder()
        framelist.append(frame)
        filelist.append(filename)
    else:
        for zmin, zmax in zbins:
            print(
                "creating redshift slice %.3f <= z < %.3f" % (zmin, zmax))
            filename = bindir.zbin_filename(
                zmin, zmax, ".fits", prefix="bin")
            if z_name is None:
                filename = bindir.zbin_filename(
                    zmin, zmax, ".fits", prefix="bin")
                os.symlink(filepath, filename)
                bindata = data
                frame = pd.DataFrame({
                    "ra": data[ra_name].byteswap().newbyteorder(),
                    "dec": data[dec_name].byteswap().newbyteorder()})
            else:
                mask = (data[z_name] > zmin) & (data[z_name] <= zmax)
                print(
                    "selected %d out of %d objects" % (
                        np.count_nonzero(mask), len(mask)))
                bindata = data[mask]
                # write the bin data to a new fits file
                hdu = pyfits.BinTableHDU(header=head, data=bindata)
                hdu.writeto(filename)
                # keep the bin data as pandas DataFrame
                frame = pd.DataFrame({
                    "ra": bindata[ra_name].byteswap().newbyteorder(),
                    "dec": bindata[dec_name].byteswap().newbyteorder(),
                    "z": bindata[z_name].byteswap().newbyteorder()})
            if weight_name is not None:
                frame["weights"] = \
                    bindata[weight_name].byteswap().newbyteorder()
            if region_name is not None:
                frame["region_idx"] = \
                    bindata[region_name].byteswap().newbyteorder()
            framelist.append(frame)
            filelist.append(filename)
    return framelist, filelist


def run_ac_single_bin(
        datapack, randpack, rlims, comoving, inv_distance_weight, R_D_ratio,
        regionize_unknown, pair_maker_instance):
    try:
        D_R_ratio = 1.0 / float(R_D_ratio)
    except ValueError:
        D_R_ratio = R_D_ratio
    zd, bindata = datapack
    zr, binrand = randpack
    est = pair_maker_instance
    est._verbose = False
    est._threads = 1
    if len(bindata) == 0 or len(binrand) == 0:
        dummy_counts = est.getDummyCounts(
            rmin=rlims[0], rmax=rlims[1], comoving=False,
            inv_distance_weight=inv_distance_weight,
            reference_weights=("weights" in bindata))
        return [est.getMeta(), dummy_counts]
    else:
        est.setRandoms(**binrand)
        est.setUnknown(**bindata)
        est.setReference(**bindata)
        est.countPairs(
            rmin=rlims[0], rmax=rlims[1], comoving=comoving,
            inv_distance_weight=inv_distance_weight,
            D_R_ratio=D_R_ratio, regionize_unknown=regionize_unknown)
        print("processed data in z ∈", zd)
        return [est.getMeta(), est.getCounts()]


def write_argparse_summary(args, outputpath, ignore=[]):
    """
    write a file summarizing all input parameters
    ARGS:    args:       argparser.Namespace object
             outputpath: file path
             ignore:     list with Namespace objects to omit
    """
    ignore = list(ignore)
    ignore.extend(["redo", "wdir"])
    arg_dict = {
        key: value for key, value in vars(args).items()
        if key not in ignore}
    with open(outputpath, 'w') as f:
        json.dump(arg_dict, f)


def load_argparse_summary(filepath):
    """
    load file produced with write_argparse_summary(), recovering original data
    type
    ARGS:    filepath: file to load
    RETURNS: dictionary with input flags
    """
    with open(filepath) as f:
        return json.load(f)
