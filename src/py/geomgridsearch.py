#!/usr/bin/env python3

import os
import subprocess
import argparse
import numpy as np

from angles2geom import (
    update_geom_params_dict,
    update_geom_params_dict,
    read_geom_to_dict,
)  # pylint: disable-all


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


def distribution2geomfiles(
    input_geometry: str, distribution: np.ndarray, folder=os.getcwd(), relative=True
) -> None:
    """
    distribution2geomfiles spawns creation of geometry fiels from a given distribution

    Parameters
    ----------
    input_geometry : str
        Input geometry file to serve as a base
    distribution : np.ndarray
        Given distribution
    folder : [bool], optional
        Folder to put all geometries in, by default os.getcwd()
    relative : [bool], optional
        Whether to invoke relative shifts
    """

    initial_geom_dict = read_geom_to_dict(input_geometry)

    for alpha, beta, corner_x, corner_y, coffset in distribution:
        new_geom_dict = update_geom_params_dict(
            initial_geom=initial_geom_dict,
            alpha=alpha,
            beta=beta,
            coffset=coffset,
            corner_x=corner_x,
            corner_y=corner_y,
            relative=relative,
        )

        new_geom_file = update_geom_file_from_dict(
            input_geometry, dict_to_apply=new_geom_dict, inplace=False
        )
        if folder != os.getcwd():
            os.mkdir(folder)
            os.system(f"mv {new_geom_file} {folder}")

        print(f"All geometry files written to {folder}")


def main(args):
    parser = argparse.ArgumentParser(
        description="Spawns many geometry files for a given ranges of parameters"
    )

    parser.add_argument(
        "input_file", type=str, help="Input geometry file to serve as template"
    )
    parser.add_argument(
        "--ralpha", type=str, default="0.0 0.0", help="Range for alpha (in degrees)"
    )
    parser.add_argument(
        "--rbeta", type=str, default="0.0 0.0", help="Range for beta (in degrees)"
    )
    parser.add_argument(
        "--rcorner_x",
        type=str,
        default="0.0 0.0",
        help="Range for corner_x (in pixels)",
    )
    parser.add_argument(
        "--rcorner_y",
        type=str,
        default="0.0 0.0",
        help="Range for corner_y (in pixels)",
    )
    parser.add_argument(
        "--rcoffset",
        type=str,
        default="0.0 0.0",
        help="Range for coffset (in meters, as in geometry file)",
    )

    parser.add_argument(
        "--relative",
        action="store_true",
        help="Whether to invoke relative shifts in corner_x, corner_y, coffset",
    )
    parser.add_argument(
        "--nsamples", type=int, help="Number of geomery files to generate"
    )
    parser.add_argument("--folder", type=str, help="Output folder to put geoms in")

    args = parser.parse_args()

    distribution = ranges2distribution(
        nsamples=args.nsamples,
        ralpha=map(float, args.ralpha.split()),
        rbeta=map(float, args.rbeta.split()),
        rcorner_x=map(float, args.rcorner_x),
        rcorner_y=map(float, args.rcorner_y),
        rcoffset=map(float, args.rcoffset),
    )
    distribution2geomfiles(
        input_geometry=args.input_file,
        distribution=distribution,
        folder=args.folder,
        relative=args.relative,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
