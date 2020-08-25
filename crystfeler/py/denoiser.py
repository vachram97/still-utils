#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import sys
from tqdm import tqdm
import os


def denoise_lst(
    input_lst: str,
    cxi_path="/entry_1/data_1/data",
    output_cxi_prefix=None,
    output_lst=None,
    compression="gzip",
    chunks=True,
    chunksize=100,
    chunksize_threshold=50,
    percentile=45,
    zero_negative=True,
) -> None:

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

    if output_cxi_prefix is None:
        output_cxi_prefix = ""

    cxi_cnt = 0
    output_cxi_list = []
    for input_cxi, events in tqdm(
        events_dict.items(), desc=f"Processing files in {input_lst} one by one"
    ):
        cxi_cnt += 1

        # create events list
        if None in events:
            with h5py.File(input_cxi, "r") as fin:
                cxi_fist_dimension = fin.shape[0]
                events = list(range(cxi_fist_dimension))
        else:
            events = sorted(list(events))

        num_images = len(events)

        with h5py.File(input_cxi, "r") as h5fin:
            data = h5fin[cxi_path]

            for chunk_idx in range(0, num_images, chunksize):
                start, stop = chunk_idx * chunksize, (chunk_idx + 1) * chunksize
                current_data = data[events][start:stop]
                if current_data.shape < chunksize_threshold:
                    print(
                        f"Some images from {input_cxi} won't be saved since their shape is less than {chunksize_threshold}"
                    )
                else:
                    new_data = process_chunk(current_data, percentile=percentile)
                    if zero_negative:
                        new_data[new_data < 0] = 0

                    output_cxi = f'{input_lst.rsplit(".")[0]}_{cxi_cnt}_{chunk_idx}.cxi'
                    shape = current_data.shape

                    with h5py.File(output_cxi, "w") as h5fout:
                        h5fout.create_dataset(
                            cxi_path,
                            shape,
                            compression="gzip",
                            data=new_data,
                            chunks=chunks,
                        )


def apply_mask(np_arr, center, r=45):

    shape = np_arr.shape[1:]
    mask = np.ones(shape)

    rx, ry = map(int, center)
    for x in range(rx - r, rx + r):
        for y in range(ry - r, ry + r):
            if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
                mask[x][y] = 0

    mask = mask.reshape((*shape, 1))
    return np_arr * mask.reshape(1, *shape)


def background(arr, q=45):
    """Return median of q-th percentile of each pixel"""

    print("Start background estimation")
    return np.percentile(arr, q=q, axis=(0))


def bin_scale(arr, b, alpha=0.01, num_iterations=10):
    """
    bin_scale binary search for proper scale factor

    Parameters
    ----------
    arr : np.ndarray
        Input 3-D array (N + 2D)
    b : background
        Single image (backgorund profile)
    alpha : float, optional
        Share of pixels to be negative, by default 0.01
    num_iterations : int, optional
        Number of binary search iterations, by default 10

    Returns
    -------
    np.ndarray
        proper scalefactors
    """

    num_negative = alpha * arr.shape[0] * arr.shape[1]

    def count_negative(scale):
        return (arr - scale * b < 0).sum()

    l, r, m = 0, 1, 2

    for _ in range(num_iterations):
        m = (l + r) / 2
        mv = count_negative(m)

        if mv < num_negative:
            l, r = m, r
        else:
            l, r = l, m

    return l


def scalefactors_bin(arr, bg, alpha=0.01, num_iterations=10):
    """\
    Find proper scalefactor for an image given a background 
    so that the share of negative pixels in resulting difference 
    is less thatn threshold
    """
    return np.array(
        [
            bin_scale(arr[i], bg, alpha=alpha, num_iterations=num_iterations)
            for i in range(arr.shape[0])
        ]
    )


def process_chunk(data, center=(720, 710), percentile=45):
    data = apply_mask(data, center=center).astype(np.int16)
    bg = background(data, q=percentile).astype(np.int16)
    scales = scalefactors_bin(data, bg, alpha=5e-2)

    full_bg = np.dot(bg.reshape(*(bg.shape), 1), scales.reshape(1, -1)).astype(np.int16)
    full_bg = np.moveaxis(full_bg, 2, 0)

    data = data - full_bg
    del full_bg
    return data


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Denoises images using simple percentile filter"
    )
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--datapath", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument("--center", type=str, help="Center position", default=None)
    parser.add_argument(
        "--out_prefix", type=str, help="Prefix for output cxi files", default=None
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Defines size of a single denoising chunk",
        default=100,
    )
    parser.add_argument(
        "--chunksize_threshold",
        type=int,
        help="Chunks less then that size wont be processed",
        default=50,
    )
    parser.add_argument(
        "--zero_negative",
        type=bool,
        help="Whether to zero out negative pixels or not",
        default=True,
    )
    parser.add_argument(
        "--out_prefix", type=str, help="Prefix for output cxi files", default=None
    )
    parser.add_argument("--percentile", type=int, help="Percentile to use", default=45)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for scale factor determination",
        default=0.05,
    )
    parser.add_argument(
        "--radius", type=int, default=45, help="Radius to apply mask in the center"
    )

    args = parser.parse_args()

    center = map(float, args.center.split())
    center = list(center)
    denoise_lst(
        input_lst,
        cxi_path=args.datapath,
        output_cxi_prefix=args.out_prefix,
        chunksize=args.chunksize,
        chunksize_theshold=args.chunksize_threshold,
        zero_negative=args.zero_negative,
        percentile=args.percentile,
    )


if __name__ == "__main__":

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
