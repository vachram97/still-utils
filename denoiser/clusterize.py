import os
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from typing import Union
from tqdm import tqdm

from imagereader import ImageBatchReader


def cluster_ndarray(
        profiles_arr,
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
    profiles = np.array([elem[1] for elem in profiles_arr])
    names = np.array([elem[0] for elem in profiles_arr])

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
        for name in names[belong_to_this_idx]:
            clusters[out_cluster_idx].add(name)
        if output_lists:
            with open(fout_name, "a") as fout:
                print(*clusters[out_cluster_idx], sep="\n", file=fout)

    if output_lists:
        return list(out_lists)
    else:
        return clusters


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
        input_lst: str, center, cxi_path="/entry_1/data_1/data", h5_path="/data/rawdata0", chunksize=100, radius=45,
):
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
    List
        [filename, *profile]
    """

    loader = ImageBatchReader(input_lst, cxi_path=cxi_path, h5_path=h5_path, chunksize=chunksize)

    profiles = []

    for lst, data in tqdm(loader, desc='Converting images to radial profiles'):

        for elem in zip(lst, data):
            profile = _radial_profile(elem[1], center=center)
            profiles.append([elem[0], profile])

    return profiles