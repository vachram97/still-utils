#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import glob
import sys


def load_lst_as_np(input_lst: str, datapath: str, top=None):
    """Loads list of images as np array"""

    if top is None:
        top = int(1e10)

    from collections import defaultdict

    images = defaultdict(lambda: set())
    with open(input_lst) as fin_lst:
        for line in fin_lst:
            imagename, eventnum = line.replace("//", "").split()
            images[imagename].add(int(eventnum))

    answ = []
    for image in images:
        idx = sorted(list(images[image]))
        full_image_path = f'{input_lst.rsplit("/", 1)[0]}/{image}'

        idx = idx[:top]

        with h5py.File(full_image_path, "r") as f:
            answ.append(np.array(f[datapath][idx]))

    return np.vstack(answ)


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


def scalefactors(arr, bkg, threshold=0.01):
    print("Scalefactors")
    scales = []
    N = arr.shape[0]
    n_pixels = arr.shape[1] * arr.shape[2]

    for idx in range(N):
        img = arr[idx]

        x = np.linspace(0, 2, 100)
        trials = [(img - bkg * scale < 0).sum() / n_pixels for scale in x]
        scale = x[np.argmin(np.array(trials) < threshold)]

        scales.append(scale)

        if idx % 10 == 0:
            print(idx)

    return np.array(scales)


def full_pipeline(arr, center):
    arr_upd = apply_mask(arr, center)

    bkg = background(arr_upd)
    scales = scalefactors(arr_upd, bkg)
    full_bkg = np.dot(bkg.reshape(*(bkg.shape), 1), scales.reshape(1, -1))
    full_bkg = np.moveaxis(full_bkg, 2, 0)

    return arr_upd - full_bkg, full_bkg


def output_arr(arr, datapath, filename='denoised.h5'):
    """Outputs array to cxi file"""

    with h5py.File(filename, "w") as f:
        f.create_dataset(datapath, data=arr)


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
        "--cxi_path", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument(
        "--center",
        type=str,
        help="Center position",
        default=None,
    )

    parser.add_argument(
        "--percentile", type=int, help="Percentile to use", default=90
    )
    parser.add_argument("--negative_threshold", type=float, help="Threshold for scale factor determination", defalut=0.01)
    parser.add_argument("--radius", type=int, default=45, help="Radius to apply mask in the center")

    args = parser.parse_args()


if __name__ == "__main__":

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
