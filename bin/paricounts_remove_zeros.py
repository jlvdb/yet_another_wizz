#!/usr/bin/env python
import argparse
import logging
import os

import tqdm
import yaw
from tqdm.contrib.logging import logging_redirect_tqdm
from yaw.core.math import array_equal


logging.basicConfig(level=logging.DEBUG)


def find_files_with_extension(directory, extension):
    files_with_extension = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                files_with_extension.append(os.path.join(root, file))
    return files_with_extension


def update_hdf_paircounts(fpath, keep=True):
    # load the original data
    old = yaw.CorrFunc.from_file(fpath)
    if keep:
        os.rename(fpath, fpath + ".old")

    # rewrite the file with new engine
    old.to_file(fpath)

    # reload the newly written data and compare the key values
    new = yaw.CorrFunc.from_file(fpath)
    for kind in ("dd", "dr", "rd", "rr"):
        old_p = getattr(old, kind)
        if old_p is None:
            continue
        new_p = getattr(new, kind)
        try:
            assert array_equal(old_p.total.totals1, new_p.total.totals1)
            assert array_equal(old_p.total.totals2, new_p.total.totals2)
            assert array_equal(old_p.count.counts, new_p.count.counts)
        except AssertionError:
            raise ValueError("converted data failed checks")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scan a directory for .hdf files containting pair count "
                    "data and rewrite them, dropping entries where all counts "
                    "are zero. (Affects files produced by yaw<2.3.1)")
    parser.add_argument(
        "path", help="directory to scan for paircount .hdf files")
    parser.add_argument(
        "--keep", action="store_true",
        help="keep a copy the original files with additional suffix '.old'")
    args = parser.parse_args()

    files = find_files_with_extension(args.path, ".hdf")
    n = len(files)
    with logging_redirect_tqdm():
        for i, path in tqdm.tqdm(enumerate(sorted(files), 1), total=n):
            update_hdf_paircounts(path, keep=args.keep)
