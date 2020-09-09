#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import os
import sys
from collections import defaultdict
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import tqdm


class ImageLoader:
    """
    ImageLoader class that abstracts the image loading process, 
    loading images one by one and returning a handle to write them
    """

    def __init__(self, input_list, chunksize=100):
        self.input = input
        self.chunksize = chunksize

        # load all frames from input list
        data = defaultdict(lambda: set())
        with open(input_list, mode="r") as fin:
            for line in fin:
                splitline = line.split()
                image = splitline[0]
                event = None if len(splitline) == 1 else splitline[1].replace("//", "")

                data[image].add(event)

        self._data = data

    def __iter__(self):
        return self

    def __next__(self, mode="r"):
        """Here the magic happens that helps to iterate"""
        """
        Pseudocode:

        data_to_return, handles_to_return = self.data[:chunksize]
        self.data = self.data - data_to_return
        return data_to_return, handles_to_return
        """


def radial_profile(data, center, normalize=True):
    """
    radial_profile returns radial profile of a 2D image

    Parameters
    ----------
    data : np.ndarray
        (M,N)-shaped np.ndarray
    center : tuple
        two float numbers as a center
    normalize : bool, optional
        whether to normalize images, by default True

    Returns
    -------
    np.ndarray
        1D numpy array
    """
    # taken from here: https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    if normalize:
        radialprofile = radialprofile / radialprofile.sum()

    return radialprofile


def lst2ndarray(
    input_lst: str, center, cxi_path="/entry_1/data_1/data", chunksize=100
) -> np.ndarray:
    """
    lst2ndarray converts CrystFEL list into 2D np.ndarray with following structure:
    np.array([filename, event_name, *profile])

    Parameters
    ----------
    input_lst : str
        input list filename
    center : [type]
        tuple of (center_x, center_y)
    cxi_path : str, optional
        datapath inside cxi/h5 file, by default "/entry_1/data_1/data"
    chunksize : int, optional
        size of chunk for reading, by default 100

    Returns
    -------
    np.ndarray
        np.array([filename, event_name, *profile])
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

    cxi_cnt = 0

    profiles_dict = {}
    for input_cxi, events in tqdm(
        events_dict.items(), desc=f"Reading cxi files from {input_lst} one by one"
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

        event_profile_pairs = []
        with h5py.File(input_cxi, "r") as h5fin:
            data = h5fin[cxi_path]

            for chunk_start in tqdm(
                range(0, num_images, chunksize), desc="Reading images in chunks"
            ):
                start, stop = chunk_start, chunk_start + chunksize

                events_idx = np.array(events)[start:stop]
                current_data = data[events_idx]
                current_data = apply_mask(current_data, center=center)

                for event, image in zip(events_idx, current_data):
                    profile = radial_profile(image, center=center)
                    event_profile_pairs.append([event, profile])

        profiles_dict[input_cxi] = event_profile_pairs

    ret = []
    for cxi_name, arr in tqdm(
        profiles_dict.items(), desc="Iterating over profiles_dict"
    ):
        for event, profile in tqdm(
            arr, desc="Iterate over each single cxi image filename"
        ):
            ret.append(np.array([cxi_name, event, profile]))
    ret = np.array(ret)
    return ret


def denoise_lst(
    input_lst: str,
    denoiser="nmf",
    cxi_path="/entry_1/data_1/data",
    output_cxi_prefix=None,
    output_lst=None,
    compression="gzip",
    chunks=True,
    chunksize=100,
    zero_negative=True,
    **denoiser_kwargs,
) -> None:

    if denoiser == "percentile":
        process_chunk = percentile_denoise
    elif denoiser == "nmf":
        process_chunk = nmf_denoise
    elif denoiser == "svd":
        process_chunk = nmf_denoise
    else:
        raise TypeError("Must provide correct denoiser")

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

            for chunk_start in tqdm(
                range(0, num_images, chunksize), desc="Running denoising in chunks"
            ):
                start, stop = chunk_start, chunk_start + chunksize
                chunk_idx = chunk_start // chunksize

                events_idx = np.array(events)[start:stop]
                current_data = data[events_idx]
                new_data = process_chunk(current_data, **denoiser_kwargs)
                if zero_negative:
                    new_data[new_data < 0] = 0

                output_cxi = f'{output_cxi_prefix}_{input_lst.rsplit(".")[0]}_{cxi_cnt}_{chunk_idx}.cxi'
                shape = current_data.shape

                with h5py.File(output_cxi, "w") as h5fout:
                    h5fout.create_dataset(
                        cxi_path,
                        shape,
                        compression="gzip",
                        data=new_data,
                        chunks=chunks,
                    )


def apply_mask(np_arr, center=(719.9, 711.5), r=45):
    """
    apply_mask applies circular mask to a single image or image series

    Parameters
    ----------
    np_arr : np.ndarray
        Input array to apply mask to
    center : tuple
        (corner_x, corner_y) pair of floats
    r : int, optional
        radius of pixels to be zeroed, by default 45

    Returns
    -------
    np.ndarray
        Same shaped and dtype'd array as input
    """

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
        return (np_arr * mask).astype(np_arr.dtype)
    else:
        mask = mask.reshape((*shape, 1))
        return (np_arr * mask.reshape(1, *shape)).astype(np_arr.dtype)


def cluster_ndarray(
    profiles_arr: np.ndarray,
    output_prefix="clustered",
    output_lists=False,
    threshold=25,
    criterion="maxclust",
    min_num_images=50,
):
    profiles = np.array([elem[2] for elem in profiles_arr])
    names_and_events = profiles_arr[:, :2]

    # this actually does clustering
    Z = ward(pdist(profiles))
    idx = fcluster(Z, t=threshold, criterion=criterion)

    # output lists
    clusters = defaultdict(lambda: [])
    for list_idx in tqdm(list(set(idx)), desc="Output lists"):
        belong_to_this_idx = np.where(idx == list_idx)[0]
        if len(belong_to_this_idx) < min_num_images:
            fout_name = f"{output_prefix}_singletone.lst"
            out_cluster_idx = -1
        else:
            fout_name = f"{output_prefix}_{list_idx}.lst"
            out_cluster_idx = list_idx

        # print output lists if you want to
        for line in names_and_events[belong_to_this_idx]:
            cxi_name, event = line
            frame_address = f"{cxi_name} //{event}"
            clusters[out_cluster_idx].append(frame_address)
        if output_lists:
            with open(fout_name, "a") as fout:
                print(clusters[out_cluster_idx], sep="\n", file=fout)

    return clusters


# -------------------------------------------
def _percentile_filter(arr, q=45):
    """Return median of q-th percentile of each pixel"""

    # print("Starf _percentile_filter estimation")
    return np.percentile(arr, q=q, axis=(0))


def bin_scale(arr, b, alpha=0.01, num_iterations=10):
    """
    bin_scale binary search for proper scale factor

    Parameters
    ----------
    arr : np.ndarray
        Input 3-D array (N + 2D)
    b background
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


def _scalefactors_bin(arr, bg, alpha=0.01, num_iterations=10):
    """\
    Find proper scalefactor for an image given background
    so that the share of negative pixels in resulting difference 
    is less thatn threshold
    """
    # print("Start scalefactor estimation")
    return np.array(
        [
            bin_scale(arr[i], bg, alpha=alpha, num_iterations=num_iterations)
            for i in range(arr.shape[0])
        ]
    )


def percentile_denoise(data, center=(720, 710), percentile=45):
    data = apply_mask(data, center=center)
    bg = _percentile_filter(data, q=percentile)
    scales = _scalefactors_bin(data, bg, alpha=5e-2)

    full_bg = np.dot(bg.reshape(*(bg.shape), 1), scales.reshape(1, -1))
    full_bg = np.moveaxis(full_bg, 2, 0)

    data = data - full_bg
    del full_bg
    return data


def nmf_denoise(arr, n_components=5, important_components=1):
    img_shape = arr.shape[1:]
    X = arr.reshape(arr.shape[0], -1)

    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    coeffs = nmf.transform(X)

    bg_full = nmf.components_
    bg_scaled = (
        coeffs[:, :important_components] @ bg_full[:important_components, :]
    ).reshape(arr.shape[0], *img_shape)

    return arr - bg_scaled


def svd_denoise(arr, n_components=5, important_components=1, n_iter=5):
    img_shape = arr.shape[1:]
    X = arr.reshape(arr.shape[0], -1)

    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=n_iter)
    svd.fit(X)
    coeffs = svd.transform(X)

    bg_full = svd.components_
    bg_scaled = (
        coeffs[:, :important_components] @ bg_full[:important_components, :]
    ).reshape(arr.shape[0], *img_shape)

    return arr - bg_scaled


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
        type=int,
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
        args.input_lst,
        cxi_path=args.datapath,
        output_cxi_prefix=args.out_prefix,
        chunksize=args.chunksize,
        chunksize_threshold=args.chunksize_threshold,
        zero_negative=args.zero_negative,
        percentile=args.percentile,
    )


# if __name__ == "__main__":
#
# import warnings
#
# with warnings.catch_warnings():
# warnings.filterwarnings("ignore", category=Warning)
# main(sys.argv[1:])
#

