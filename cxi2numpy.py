#!/usr/bin/env python3

import h5py
import numpy as np
from tqdm import tqdm
from itertools import chain, product
import matplotlib.pyplot as plt
from typing import Dict


def _peaks_from_ndarray(
    nparr: np.ndarray, size=1
) -> [np.ndarray, np.ndarray, np.ndarray]:
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

    shifts_lst = product(range(-size, size + 1), repeat=2)
    num_neighbours = (2 * size + 1) ** 2 - 2

    for roll_combination in shifts_lst:
        ret += (np.roll(nparr, roll_combination, axis=(-2, -1)) < nparr).astype(np.int8)

    return np.nonzero(ret - num_neighbours > 0)


def peak_profiles(nparray: np.ndarray, size_compare=1, size_return=3) -> list:
    """
    Returns peak profiles as (n, x, y) tensor, with n=nparray.shape[0], and x=y=2*size+1
    
    Arguments:
        nparray {np.ndarray} -- input numpy array with full images
    
    Returns:
        dict -- peak profiles
    """

    pks = _peaks_from_ndarray(nparray, size=size_compare)
    answ = {}

    if len(nparray.shape) == 3:
        for n, x, y in zip(*pks):
            profile = nparray[n][
                x - size_return : x + size_return + 1,
                y - size_return : y + size_return + 1,
            ]
            answ[(x, y)] = profile
    else:
        for x, y in zip(*pks):
            profile = nparray[
                x - size_return : x + size_return + 1,
                y - size_return : y + size_return + 1,
            ]
            answ[(x, y)] = profile

    return answ


def extract_hits(
    cxi_filename: str,
    indices: list,
    cxi_path="entry_1/data_1/data",
    size_compare=1,
    size_return=5,
) -> Dict:
    """
    From single cxi file and list of indices within it (which are usually "Event" field in crystfel stream), returns np.ndarray representing each hit image

    Arguments:
        cxi_filename {str} -- input cxi filename
        cxi_path {str} -- path to data in hierarchical cxi structure
        indices {list} -- indices of hit images along time coordinate
        size_compare {int} -- peaks should be larger or equal to all peaks within (-size,+size) square
        size_return {int} -- will return (2*size_return-1)**2 sized square with peak in center

    Returns:
        Dict -- dictionary of f'{image_filename|event : {'x':x, 'y':y, 'profile':np.ndarray}}
    """

    hits = {}
    answ = {}

    with h5py.File(cxi_filename, "r") as f:
        for idx in indices:
            current_key = (cxi_filename, idx)
            current_nparray = np.array(f[cxi_path][idx])
            hits[current_key] = current_nparray

            peaks = peak_profiles(
                current_nparray, size_compare=size_compare, size_return=size_return
            )

            size = 2*size_return + 1
            for (x, y), profile in tqdm(peaks.items()):
                if profile.shape == (size, size):
                    answ[(*current_key, x, y)] = list(profile.flatten())

    return answ


indices = [
    int(i)
    for i in [
        "5398",
        "9958",
        "5574",
        "2702",
        "9270",
        "6242",
        "5885",
        "6367",
        "7444",
        "5851",
        "8477",
        "9686",
        "6931",
        "9697",
        "6042",
        "8964",
        "8906",
        "5313",
        "6137",
        "5754",
        "9442",
        "5557",
        "9944",
        "5675",
        "8717",
        "5816",
        "5919",
        "7315",
        "8245",
        "9863",
        "8267",
        "6187",
        "5829",
        "9855",
        "6126",
        "5345",
        "6986",
        "8997",
        "6319",
        "9515",
        "9627",
        "7200",
        "6946",
        "5235",
        "5795",
        "6940",
        "6112",
        "5488",
        "7295",
        "9571",
        "5312",
        "6203",
        "6243",
    ]
]

