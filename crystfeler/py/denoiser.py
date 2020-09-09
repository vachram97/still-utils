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
from typing import Union


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


def _apply_mask(np_arr, center=(719.9, 711.5), radius=45):
    """
    _apply_mask applies circular mask to a single image or image series

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
    r = radius
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
    """
    cluster_ndarray clusters images based on their radial profiles

    Parameters
    ----------
    profiles_arr : np.ndarray
        radial profiles (or any other profiles, honestly) 2D np.ndarray
    output_prefix : str, optional
        output prefix for image lists0, by default "clustered"
    output_lists : bool, optional
        whether to output lists as text fiels, by default False
    threshold : int, optional
        distance according to criterion, by default 25
    criterion : str, optional
        criterion for clustering, by default "maxclust"
    min_num_images : int, optional
        minimal number of images in single cluster, others will go to singletone, by default 50

    Returns
    -------
    Union[dict, list]
        Either:
           - Dictionary {cluster_num:[*image_and_event_lines]} -- if output_lists == False
           - List [output_list_1.lst, output_list_2.lst, ...] -- if output_lists == True
    """
    profiles = np.array([elem[2] for elem in profiles_arr])
    names_and_events = profiles_arr[:, :2]

    # this actually does clustering
    Z = ward(pdist(profiles))
    idx = fcluster(Z, t=threshold, criterion=criterion)

    # output lists
    clusters = defaultdict(lambda: set())
    out_lists = set()
    for list_idx in tqdm(list(set(idx)), desc="Output lists"):
        belong_to_this_idx = np.where(idx == list_idx)[0]
        if len(belong_to_this_idx) < min_num_images:
            fout_name = f"{output_prefix}_singletone.lst"
            out_cluster_idx = -1
        else:
            fout_name = f"{output_prefix}_{list_idx}.lst"
            out_cluster_idx = list_idx
        out_lists.add(fout_name)
        try:
            os.remove(fout_name)
        except OSError:
            pass

        # print output lists if you want to
        for line in names_and_events[belong_to_this_idx]:
            cxi_name, event = line
            frame_address = f"{cxi_name} //{event}"
            clusters[out_cluster_idx].add(frame_address)
        if output_lists:
            with open(fout_name, "a") as fout:
                print(*clusters[out_cluster_idx], sep="\n", file=fout)

    if output_lists:
        return list(out_lists)
    else:
        return clusters


def _percentile_filter(arr, q=45):
    """
    _percentile_filter creates background profile

    Parameters
    ----------
    arr : 3D np.ndarray (series of 2D images)
        input array
    q : int, optional
        percentile for filtering, by default 45

    Returns
    -------
    np.ndarray
        2D np.ndarray of background profile
    """

    return np.percentile(arr, q=q, axis=(0))


def _bin_scale(arr, b, alpha=0.01, num_iterations=10):
    """
    _bin_scale binary search for proper scale factor

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
    Find proper scalefactor for an image given f _percentile_filter 
    so that the share of negative pixels in resulting difference 
    is less thatn threshold
    """
    # print("Start scalefactor estimation")
    return np.array(
        [
            _bin_scale(arr[i], bg, alpha=alpha, num_iterations=num_iterations)
            for i in range(arr.shape[0])
        ]
    )


def _radial_profile(data, center, normalize=True):
    """
    _radial_profile returns radial profile of a 2D image

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


def lst2profiles_ndarray(
    input_lst: str, center, cxi_path="/entry_1/data_1/data", chunksize=100, radius=45,
) -> np.ndarray:
    """
    lst2profiles_ndarray converts CrystFEL list into 2D np.ndarray with following structure:
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
                current_data = _apply_mask(current_data, center=center, radius=radius)

                for event, image in zip(events_idx, current_data):
                    profile = _radial_profile(image, center=center)
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
    center=None,
    radius=None,
    denoiser="nmf",
    cxi_path="/entry_1/data_1/data",
    output_cxi_prefix=None,
    output_lst=None,
    compression="gzip",
    chunks=True,
    chunksize=100,
    zero_negative=True,
    dtype=np.int16,
    **denoiser_kwargs,
) -> None:
    """
    denoise_lst applies denoiser to a list

    Parameters
    ----------
    input_lst : str
        input list in CrystFEL format
    denoiser : str, optional
        denoiser type, by default "nmf"
    cxi_path : str, optional
        path inside a cxi file, by default "/entry_1/data_1/data"
    output_cxi_prefix : [type], optional
        prefix for output cxi files, by default None
    output_lst : [type], optional
        output list filename, by default None
    compression : str, optional
        which losless compression to use, by default "gzip"
    chunks : bool, optional
        whether to output in chunks (saves RAM), by default True
    chunksize : int, optional
        chunksize for reading, by default 100
    zero_negative : bool, optional
        whether to convert negative values to 0, by default True

    Raises
    ------
    TypeError
        If denoiser is not in ('percentile', 'nmf','svd')
    """

    if denoiser == "percentile":
        process_chunk = percentile_denoise
    elif denoiser == "nmf":
        process_chunk = nmf_denoise
    elif denoiser == "svd":
        process_chunk = svd_denoise
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
                new_data = process_chunk(
                    current_data, center=center, radius=radius, **denoiser_kwargs
                ).astype(dtype)
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


def percentile_denoise(data, center, percentile=45, alpha=5e-2, radius=45):
    """
    percentile_denoise applies percentile denoising:
    - create percentile-based background profille
    - apply mask
    - subtract background with such scale that less thatn `alpha` resulting pixels are negative

    Parameters
    ----------
    data : np.ndarray
        Input data (series of 2D images, 3D total)
    center : tuple, optional
        (corner_x, corner_y), by default (720, 710)
    percentile : int, optional
        percentile to use, by default 45

    Returns
    -------
    np.ndarray
        Denoised images
    """
    data = _apply_mask(data, center=center, radius=radius)
    bg = _percentile_filter(data, q=percentile)
    scales = _scalefactors_bin(data, bg, alpha=alpha)

    full_bg = np.dot(bg.reshape(*(bg.shape), 1), scales.reshape(1, -1))
    full_bg = np.moveaxis(full_bg, 2, 0)

    data = data - full_bg
    del full_bg
    return data


def nmf_denoise(arr, center, n_components=5, important_components=1, radius=45):
    """
    nmf_denoise performs NMF-decomposition based denoising
    - (N, M, M) image series --> (N, M**2) flattened images
    - (N, M**2) = (N, n_components) @ (n_components, M**2) NMF decomposition
    - background: (n_components, M**2) --> (important_components, M**2)
    - scales: (N, n_components) --> (N, important_components)
    - scaled_background = scales @ background
    - return arr - scaled_background

    Parameters
    ----------
    arr : np.ndarray
        Input data (series of 2D images, 3D total)
    center : tuple
        (corner_x, corner_y) tuple
    n_components : int, optional
        n_components for dimensionality reduction, by default 5
    important_components : int, optional
        number of components to account for, by default 1

    Returns
    -------
    np.ndarray
        Denoised data
    """
    img_shape = arr.shape[1:]
    X = arr.reshape(arr.shape[0], -1)

    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    coeffs = nmf.transform(X)

    bg_full = nmf.components_
    bg_scaled = (
        coeffs[:, :important_components] @ bg_full[:important_components, :]
    ).reshape(arr.shape[0], *img_shape)

    return _apply_mask(arr - bg_scaled, radius=radius, center=center)


def svd_denoise(
    arr, center, n_components=5, important_components=1, n_iter=5, radius=45
):
    """
    svd_denoise performs SVD-based denoising of input array
    - (N, M, M) image series --> (N, M**2) flattened images
    - (N, M**2) = (N, n_components) @ (n_components, M**2) SVD decomposition
    - background: (n_components, M**2) --> (important_components, M**2)
    - scales: (N, n_components) --> (N, important_components)
    - scaled_background = scales @ background
    - return arr - scaled_background

    Parameters
    ----------
    arr : np.ndarra
        3D numpy array (series of 2D images)
    center : tuple
        (corner_x, corner_y) tuple
    n_components : int, optional
        n_components for TruncatedSVD decomposition, by default 5
    important_components : int, optional
        number of components to account fo, by default 1
    n_iter : int, optional
        number of iterations in TruncatedSVD, by default 5

    Returns
    -------
    np.ndarray
        Denoised array of same shape
    """
    img_shape = arr.shape[1:]
    X = arr.reshape(arr.shape[0], -1)

    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=n_iter)
    svd.fit(X)
    coeffs = svd.transform(X)

    bg_full = svd.components_
    bg_scaled = (
        coeffs[:, :important_components] @ bg_full[:important_components, :]
    ).reshape(arr.shape[0], *img_shape)

    return _apply_mask(arr - bg_scaled, radius=radius, center=center)


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
        "--min_num_images",
        type=int,
        help="Chunks less then that size will be sent to singletone",
        default=50,
    )
    parser.add_argument(
        "--zero_negative",
        type=int,
        help="Whether to zero out negative pixels or not",
        default=True,
    )
    parser.add_argument(
        "--out_prefix", type=str, help="Prefix for output cxi files", default="denoised"
    )
    parser.add_argument("--percentile", type=int, help="Percentile to use", default=45)
    parser.add_argument(
        "--alpha",
        type=float,
        help="Threshold for scale factor determination",
        default=0.05,
    )
    parser.add_argument(
        "--radius", type=int, default=45, help="Radius to apply mask in the center"
    )

    list_of_denoisers = ["percentile", "nmf", "svd"]
    parser.add_argument(
        "--denoiser", choices=list_of_denoisers, help="Type of denoiser to use"
    )
    parser.add_argument(
        "--output_lst_prefix",
        type=str,
        help="List prefix for lists after clustering",
        default="clustered",
    )
    parser.add_argument(
        "--clustering_distance",
        type=float,
        help="Clustering distance threshold",
        default=25.0,
    )

    parser.add_argument(
        "--criterion",
        type=str,
        help="Clustering criterion (google `scipy.fcluster`)",
        default="maxclust",
    )

    args = parser.parse_args()

    center = map(float, args.center.split())
    center = list(center)

    profiles_ndarray = lst2profiles_ndarray(
        args.input_lst, center=center, cxi_path=args.datapath, chunksize=args.chunksize
    )
    image_lists_list = cluster_ndarray(
        profiles_ndarray,
        output_prefix=args.output_lst_prefix,
        output_lists=True,
        threshold=args.clustering_distance,
        criterion=args.criterion,
        min_num_images=args.min_num_images,
    )

    for list_to_denoise in tqdm(image_lists_list, desc="Denoising each clustered list"):
        denoise_lst(
            input_lst=list_to_denoise,
            center=center,
            denoiser=args.denoiser,
            cxi_path=args.datapath,
            output_cxi_prefix=args.out_prefix,
            chunksize=args.chunksize,
            zero_negative=args.zero_negative,
            radius=args.radius,
        )


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])

