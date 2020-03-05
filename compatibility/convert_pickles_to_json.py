#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

from yet_another_wizz.utils import dump_json


parser = argparse.ArgumentParser(
    description='Convert a region count or bin weight dictionary stored as '
                'python3 pickle to JSON and rename keys to new schema.')
parser.add_argument(
    'pklfile', help='input pickle file')
parser.add_argument(
    '-d', '--delete', action='store_true', help='delete input file')
parser.add_argument(
    '-s', '--show', action='store_true',
    help='show the output file (disables --delete)')


if __name__ == "__main__":

    args = parser.parse_args()

    try:
        with open(args.pklfile, "rb") as f:
            data_dict = pickle.load(f)
        # remap the keys
        data_dict["sum_redshifts"] = data_dict.pop("redshift")
        data_dict["data_data"] = data_dict.pop("unknown")
        data_dict["data_random"] = data_dict.pop("rand")
        if "amplitude_factor" in data_dict:
            data_dict["width_correction"] = \
                data_dict.pop("amplitude_factor")
    except KeyError:
        pass
    except Exception as e:
        sys.exit("ERROR: input is not a valid dictionary pickle")
    # convert to JSON
    output = os.path.splitext(args.pklfile)[0] + ".json"
    dump_json(data_dict, output, preview=args.show)
    if args.delete and not args.show:
        os.remove(args.pklfile)
