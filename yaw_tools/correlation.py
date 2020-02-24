import os

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits


def bin_table(
        bindir, filepath, ra_name, dec_name, z_name, weightname=None,
        zbins=None, cat_ext=1):
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
                "RA": data[ra_name].byteswap().newbyteorder(),
                "DEC": data[dec_name].byteswap().newbyteorder()})
        else:
            zmin, zmax = data[z_name].min(), data[z_name].max()
            filename = bindir.zbin_filename(zmin, zmax, ".fits", prefix="bin")
            os.symlink(filepath, filename)
            frame = pd.DataFrame({
                "RA": data[ra_name].byteswap().newbyteorder(),
                "DEC": data[dec_name].byteswap().newbyteorder(),
                "z": data[z_name].byteswap().newbyteorder()})
        if weightname is not None:
            frame["weight"] = data[weightname].byteswap().newbyteorder()
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
                frame = pd.DataFrame({
                    "RA": data[ra_name].byteswap().newbyteorder(),
                    "DEC": data[dec_name].byteswap().newbyteorder()})
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
                    "RA": bindata[ra_name].byteswap().newbyteorder(),
                    "DEC": bindata[dec_name].byteswap().newbyteorder(),
                    "z": bindata[z_name].byteswap().newbyteorder()})
            if weightname is not None:
                frame["weight"] = \
                    bindata[weightname].byteswap().newbyteorder()
            framelist.append(frame)
            filelist.append(filename)
    return framelist, filelist


def get_bin_weights(framelist, filelist):
    weight_dict = {}
    for frame, path in zip(framelist, filelist):
        try:
            weight = frame.weight.sum()
        except AttributeError:  # assume uniform weights
            weight = len(frame)
        key = os.path.basename(path)  # bin_{:.3f}z{:.3f}.fits
        key = os.path.splitext(key)[0]  # bin_{:.3f}z{:.3f}
        key = key.strip("bin_")  # {:.3f}z{:.3f}
        weight_dict[key] = weight
    return weight_dict


def run_ac_single_bin(
        datapack, randpack, rlims, R_D_ratio, regionize_unknown,
        pm_instance):
    try:
        D_R_ratio = 1.0 / float(R_D_ratio)
    except ValueError:
        D_R_ratio = R_D_ratio
    zd, bindata = datapack
    zr, binrand = randpack
    est = pm_instance
    est._verbose = False
    est._threads = 1
    if len(bindata) == 0 or len(binrand) == 0:
        dummy_counts = est.getDummyCounts(
            rmin=rlims[0], rmax=rlims[1], comoving=False,
            reference_weights=("weights" in bindata))
        return [est.getMeta(), dummy_counts]
    else:
        try:
            est.setUnknown(bindata.RA, bindata.DEC, bindata.z, bindata.weight)
        except AttributeError:
            est.setUnknown(bindata.RA, bindata.DEC, bindata.z)
        try:
            est.setRandoms(binrand.RA, binrand.DEC, binrand.z, binrand.weight)
        except AttributeError:
            est.setRandoms(binrand.RA, binrand.DEC, binrand.z)
        est.setReference(bindata.RA, bindata.DEC, bindata.z)
        est.countPairs(
            rmin=rlims[0], rmax=rlims[1], comoving=False,
            D_R_ratio=D_R_ratio, regionize_unknown=regionize_unknown)
        print("processed data in z ∈", zd)
        return [est.getMeta(), est.getCounts()]
