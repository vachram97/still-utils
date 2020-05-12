#!/usr/bin/env python3

import argparse
import h5py
import os
from tqdm import tqdm
import numpy as np
import sys


def lst2cxi(
    input_lst: str, cxi_path="/entry_1/data_1/data", output_cxi=None, compression="gzip"
) -> str:
    """
    Saves all images from input cxi file that are present in input list.

    Arguments:
        input_lst {str} -- input list (might be both with and without events)
        cxi_path {str} -- path of data inside cxi file to be copied
        output_cxi {None} -- output cxi file. If 'None', will be <input_list>.cxi for <input_list>.lst

    Returns:
        str -- filename of cxi file
    """

    events_dict = {}
    with open(input_lst, "r") as fin:
        for line in fin:
            if "//" in line:  # if line represents an event in cxi
                filename, event = line.split(" //")
                event = int(event)
                if filename[0] != "/":  # if path is not absolute
                    filename = f"{os.getcwd()}/{filename}"
                if filename in events_dict:
                    if None not in events_dict[filename]:
                        events_dict[filename].add(event)
                    else:
                        pass
                else:
                    events_dict[filename] = {event}
            else:  # if line represents both event and its absence
                filename = line.rstrip()
                if filename[0] != "/":  # if path is not absolute
                    filename = os.getcwd() + filename
                events_dict[line] = {None}

    if output_cxi is None:
        output_cxi = input_lst.rsplit(".")[0] + ".cxi"

    with h5py.File(output_cxi, "w") as h5fout:
        for input_cxi, events in events_dict.items():
            with h5py.File(input_cxi, "r") as h5fin:
                if None in events:  # meaning we must dump whole cxi
                    data = h5fin[cxi_path]
                    shape = data.shape
                    h5fout.create_dataset(
                        cxi_path, shape, compression="gzip", data=data.value
                    )
                else:
                    print(h5fin.keys())
                    print(cxi_path)
                    data = h5fin[cxi_path]
                    data = np.array(data.value)[np.array(list(events))]
                    shape = data.shape
                    h5fout.create_dataset(
                        cxi_path, shape, compression="gzip", data=data
                    )

    return output_cxi


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Will save only necessary files from input list"
    )
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--cxi_path", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument(
        "--output_cxi",
        type=str,
        help="Output cxi file (if not present, will replace input *.lst to *.h5) ",
        default=None,
    )

    args = parser.parse_args()

    if args.cxi_path is None:
        ret = lst2cxi(input_lst=args.input_lst, output_cxi=args.output_cxi)
    else:
        ret = lst2cxi(
            input_lst=args.input_lst, cxi_path=args.cxi_path, output_cxi=args.output_cxi
        )

    print(f"Data successfully written to {ret}")


if __name__ == "__main__":
    main(sys.argv[1:])
