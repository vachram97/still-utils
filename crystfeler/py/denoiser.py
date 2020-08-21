#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import glob
import sys


def load_cxi(input_cxi: str, datapath: str, top=None):
    """Loads a cxi image as np array"""

    if top is None:
        top = int(1e10)

    with h5py.File(input_cxi, "r") as f:
        ret = np.array(f[datapath][:top])

    return ret


def apply_mask(np_arr, center, r=45):

    print("Mask")

    if len(np_arr.shape) == 3:
        shape = np_arr.shape[1:]
        shape_type = 3
    else:
        shape = np_arr.shape
        shape_type = 2
    mask = np.ones(shape)

    rx, ry = map(int, center)
    for x in range(rx - r, rx + r):
        for y in range(ry - r, ry + r):
            if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
                mask[x][y] = 0

    if shape_type == 2:
        return np_arr * mask
    else:
        mask = mask.reshape((*shape, 1))
        return np_arr * mask.reshape(1, *shape)


def background(arr, q=90):
    """Return median of q-th percentile of each pixel"""

    p = np.percentile(arr, q=90, axis=(0))
    b = np.median(arr * (arr < p), axis=0)

    return b


def scalefactors(arr, bkg, threshold=0.01, nsteps=100):
    """\
    Find proper scalefactor for an image given a background 
    so that the share of negative pixels in resulting difference 
    is less thatn threshold
    """

    print("Scalefactors")
    scales = []
    N = arr.shape[0]
    n_pixels = arr.shape[1] * arr.shape[2]

    for idx in range(N):
        img = arr[idx]

        x = np.linspace(0, 2, nsteps)
        trials = [(img - bkg * scale < 0).sum() / n_pixels for scale in x]
        scale = x[np.argmin(np.array(trials) < threshold)]

        scales.append(scale)

        if idx % 10 == 0:
            print(idx)

    return np.array(scales)


def output_arr(arr, datapath, filename="denoised.h5"):
    """Outputs array to cxi file"""

    with h5py.File(filename, "w") as f:
        f.create_dataset(datapath, data=arr)


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Denoises images using simple percentile filter"
    )
    parser.add_argument(
        "input_cxi",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--datapath", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument("--center", type=str, help="Center position", default=None)

    parser.add_argument("--percentile", type=int, help="Percentile to use", default=90)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for scale factor determination",
        default=0.01,
    )
    parser.add_argument(
        "--radius", type=int, default=45, help="Radius to apply mask in the center"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=200,
        help="Number of points between 0 and 2 to try determine a scalefactor for each image",
    )

    args = parser.parse_args()

    arr = load_cxi(args.input_cxi, datapath=args.datapath)
    center = map(float, args.center.split())
    center = list(center)

    arr_upd = apply_mask(arr, center)

    bkg = background(arr_upd, q=args.percentile)
    scales = scalefactors(arr_upd, bkg, threshold=args.threshold, nsteps=args.nsteps)
    full_bkg = np.dot(bkg.reshape(*(bkg.shape), 1), scales.reshape(1, -1))
    full_bkg = np.moveaxis(full_bkg, 2, 0)

    denoised = arr_upd - full_bkg

    denoised_filename = f"{args.input_cxi.rsplit('.')}_denoised.cxi"
    bkg_filename = f"{args.input_cxi.rsplit('.')}_background.cxi"
    output_arr(denoised, datapath=args.datapath, filename = denoised_filename)
    output_arr(full_bkg, datapath=args.datapath, filename = bkg_filename)


if __name__ == "__main__":

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
