#!/usr/bin/env python3

import os
import subprocess
import argparse
import numpy as np

# from .angles2geom import read_geom_to_dict, update_geom_file_from_dict, update_geom_params_dict


def ranges2distribution(
    nsamples: int,
    ralpha=(0.0, 0.0),
    rbeta=(0.0, 0.0),
    rcorner_x=(0, 0),
    rcorner_y=(0, 0),
    rcoffset=(0, 0),
) -> np.ndarray:
    """
    ranges2distribution generates sample points in a given range

    Parameters
    ----------
    ralpha : tuple, optional
        range for alpha (in degrees), by default (0.0, 0.0)
    rbeta : tuple, optional
        range for beta (in degrees) by default (0.0, 0.0)
    rcorner_x : tuple, optional
        range for corner_x (in pixels) by default (0, 0)
    rcorner_y : tuple, optional
        range for corner_y (in pixels) by default (0, 0)
    rcoffset : tuple, optional
        range for coffset (in meters) by default (0, 0)

    Returns
    -------
    np.ndarray
        (nsamples, 5)-shape np.ndarray of uniformly distributed points in given ranges
    """

    args = [ralpha, rbeta, rcorner_x, rcorner_y, rcoffset]
    low = [elem[0] for elem in args]
    high = [elem[1] for elem in args]
    size = (len(args),)

    ret = np.array(
        [np.random.uniform(low=low, high=high, size=size) for _ in range(nsamples)]
    )

    return ret


def distribution2geomfiles(input_geometry: str, distribution:np.ndarray, folder=os.getcwd()) -> None:
    """
    distribution2geomfiles spawns creation of geometry fiels from a given distribution

    Parameters
    ----------
    input_geometry : str
        Input geometry file to serve as a base
    distribution : np.ndarray
        Given distribution
    folder : [type], optional
        Folder to put all geometries in, by default os.getcwd()
    """

    pass