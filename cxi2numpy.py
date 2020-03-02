#!/usr/bin/env python3

import h5py
import numpy as np
from itertools import chain, product
import matplotlib.pyplot as plt


def extract_hits(
    cxi_filename: str, cxi_path: str, indices: list, prefix="png", plot=True, **args
) -> np.ndarray:
    """
    From single cxi file and list of indices within it (which are usually "Event" field in crystfel stream), returns np.ndarray representing each hit image

    Arguments:
        cxi_filename {str} -- input cxi filename
        cxi_path {str} -- path to data in hierarchical cxi structure
        indices {list} -- indices of hit images along time coordinate

    Keyword Arguments:
        prefix {str} -- if 'plot==True', where to save png pictures of hits (default: {'png'})
        plot {bool} -- whether to plot the hits as pngs (default: {True})

    Returns:
        np.ndarray -- (n_images, xsize, ysize), containing hits only
    """

    data = []

    with h5py.File(cxi_filename, "r") as f:
        for idx in indices:
            current_nparray = np.array(f["entry_1/data_1/data"][idx])
            data.append(current_nparray)

            if plot:
                plt.imshow(current_nparray, **args)
                plt.colorbar()
                plt.savefig(f"{prefix}/{cxi_filename}_{idx}.png", dpi=300)

    return np.array(data)


def peaks_from_ndarray(nparr: np.ndarray, size=1) -> [np.ndarr, np.ndarr, np.ndarr]:
    """
    Returns positions of peaks -- pixels that are larger than any of their neighbours within a certain radius
    
    Arguments:
        ndarr {np.ndarray} -- input array. Last two coordinates are treated as `x` and `y`.
        size {int} -- size of square of neighbours will be (2n+1)
    
    Returns:
        [np.ndarr, np.ndarr, np.ndarr] -- return value of np.nonzero. Shape corresponds to input array shape
    """
    ...
    ret = np.zeros(nparr.shape).astype(np.int8)

    shifts_lst = product([range(-size, size + 1)], repeat=2)
    for roll_combination in shifts_lst:
        ret += (np.roll(nparr, roll_combination, axis=(0, 1)) < nparr).astype(np.int8)

    return np.nonzero(ret)


def peak_profiles(nparray: np.ndarray, size=3) -> np.ndarray:
    """
    Returns peak profiles as (n, x, y) tensor, with n=nparray.shape[0], and x=y=2*size+1
    
    Arguments:
        nparray {np.ndarray} -- input numpy array with full images
    
    Returns:
        np.ndarray -- peak profiles
    """

    pks = peaks_from_ndarray(nparray)
    answ = []

    if len(nparray.shape) == 3:
        ns, xs, ys = pks
        for n, x, y in pks:
            profile = nparray[n][x-size:x+size+1, y-size:y+size+1]
            answ.append(profile)
