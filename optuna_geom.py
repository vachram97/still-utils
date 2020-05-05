#!/usr/bin/env python3

import argparse
import hyperopt
import sys
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List


def run_crystfel(input_yaml_file: str) -> str:
    """
    Runs crystfel with a certain input yaml file and gets a name of the output folder
    Assumes that stream is linked to `./laststream`, as is usually happens.

    Arguments:
        input_yaml_file {str} -- input file with crystfel parameters (for `runner`)

    Returns:
        subprocess.Process -- runner process
    """
    runner = subprocess.run(f"runner {input_yaml_file}", stdout=subprocess.PIPE)
    return runner


def num_crystals(input_stream="./laststream") -> int:
    """
    Counts number of indexed crystals

    Keyword Arguments:
        input_stream {str} -- input stream filename (default: {'./laststream'})

    Returns:
        int -- number of crystals in stream
    """
    num_crystals = subprocess.run(
        f"grep -c 'Begin crystal' {input_stream}", stdout=subprocess.PIPE
    )
    if num_crystals.check_returncode():
        return None
    return int(num_crystals.stdout.decode())


def update_yaml(in_yaml: str, geom_params: dict, inplace=True) -> str:
    """
    Updates yaml file with given geometry params

    Arguments:
        in_yaml {str} -- input yaml filename
        geom_params {dict} -- geometry parameters in crystfel format

    Returns:
        str -- updated yaml path
    """

    with open(in_yaml, "r") as stream:
        yaml_params = stream.read().split("\n")
    in_geom, geom_line_index = [
        (line.split(": ")[-1], idx)
        for idx, line in enumerate(yaml_params)
        if line.startswith("GEOM: ")
    ][0]

    out_yaml = in_yaml if inplace else f"{in_yaml}.upd"
    out_geom = in_geom if inplace else f"{in_geom}.upd"

    with open(in_geom, "r") as fin:
        geom_lines = fin.read().split("\n")

    # update geometry with geom_params
    for idx, line in enumerate(geom_lines):
        for key, value in geom_params.items():
            if key in line:
                line = " = ".join(map(str, (key, value)))
        geom_lines[idx] = line

    with open(f"{out_geom}", "w") as geom_fout:
        print(*geom_lines, file=geom_fout, sep="\n")

    yaml_params[geom_line_index] = yaml_params[geom_line_index].replace(
        in_geom, out_geom
    )
    with open(out_yaml, "w") as fout:
        print(*yaml_params, sep="\n", file=fout)

    return out_yaml


def update_geom_params(
    initial_geom,
    alpha: np.float,
    beta: np.float,
    coffset: np.float,
    corner_x: np.float,
    corner_y: np.float,
) -> dict:
    """
    Given an initial dict of values, 
    generates dict of geometry params, 
    given certain alpha, beta, coffset, corner_x and corner_y

    Arguments:
        initial_dict {[type]} -- initial dictionary, containing all the values
        alpha {np.float} -- angle of rotation along vertical axis (deg)
        beta {np.float} -- angle of rotation along horizontal axis (deg)
        coffset {np.float} -- crystal-to-detector length offset
        corner_x {np.float} -- x coordinate of detector corner
        corner_y {np.float} -- y coordinate of detector corner

    Returns:
        dict -- updated dictioinary
    """

    important_keys = ["fs", "ss", "corner_x", "corner_y", "coffset"]
    initial_dict = {}
    with open(initial_geom, "r") as fin:
        for line in fin:
            for key in important_keys:
                if line[0] != ";" and line.replace(" ", "").split("=")[0].endswith(
                    f"/{key}"
                ):
                    key, value = line[:-1].replace(" ", "").split("=")
                    initial_dict[key] = value

    print(initial_dict)
    alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)

    def _find_proper_key(val: str):
        for key in initial_dict:
            if key.endswith(val):
                return key

    fs_key, ss_key, corner_x_key, corner_y_key, coffset_key = map(
        _find_proper_key, important_keys
    )

    # p0/fs = -1.000000x +0.000000y
    # p0/ss = +0.000000x -1.000000y
    # p0/corner_x = 719.846731
    # p0/corner_y = 711.369219
    # p0/coffset = 0.0

    answ = initial_dict.copy()
    answ[corner_x_key] = corner_x
    answ[corner_y_key] = corner_y
    answ[coffset_key] = coffset

    fs, ss = initial_dict[fs_key], initial_dict[ss_key]

    fs = f"{fs} + 0.0z" if "z" not in fs else fs
    ss = f"{ss} + 0.0z" if "z" not in ss else ss

    M = angles2matrix(fs, ss)
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])

    rotation_z = R.from_rotvec(alpha * z_axis)
    rotation_y = R.from_rotvec(beta * y_axis)
    M = rotation_y.apply(rotation_z.apply(M))

    coordinates = ("x", "y", "z")
    answ[fs_key] = " ".join(
        [f"{elem:+f}{name}" for elem, name in zip(M[0], coordinates)]
    )
    answ[ss_key] = " ".join(
        [f"{elem:+f}{name}" for elem, name in zip(M[1], coordinates)]
    )

    return answ


def angles2matrix(ss_line: str, fs_line: str):

    fs_line = f"{fs_line} + 0.0z" if "z" not in fs_line else fs_line
    ss_line = f"{ss_line} + 0.0z" if "z" not in ss_line else ss_line

    std = lambda s: s.replace(" ", "")

    assert list(sorted(filter(str.isalpha, std(fs_line)))) == [
        "x",
        "y",
        "z",
    ], "Please change fs line to follow alphabetic x, y, z order"
    assert list(sorted(filter(str.isalpha, std(ss_line)))) == [
        "x",
        "y",
        "z",
    ], "Please change ss line to follow alphabetic x, y, z order"

    x1, *rest1 = std(fs_line).split("x")
    y1, *rest1 = rest1[0].split("y")
    z1, *rest1 = rest1[0].split("z")

    x2, *rest2 = std(ss_line).split("x")
    y2, *rest2 = rest2[0].split("y")
    z2, *rest2 = rest2[0].split("z")

    M = np.array([[x1, y1, z1], [x2, y2, z2]]).astype(np.float)
    return M


def main(args: List[str]):

    parser = argparse.ArgumentParser(
        description="hyperopt-based optimisator of single-panel CrystFEL geometry files"
    )
    parser.add_argument("yaml", type=str, help="Input YAML file for runner")
    parser.add_argument(
        "--alpha",
        type=str,
        help='Range along alpha (z-axis) angle, "-1.0 1.0" would be [-1,1] degree',
    )
    parser.add_argument(
        "--beta",
        type=str,
        help='Range along beta (y-axis) angle, "-1.0 1.0" would be [-1,1] degree',
    )
    parser.add_argument("--corner_x", type=str, help="Range for corner_x")
    parser.add_argument("--corner_y", type=str, help="Range for corner_y")
    parser.add_argument("--coffset", type=str, help="Range for coffset")
    parser.add_argument("--ntrials", type=int, help="Number of trials for optimization")

    args = parser.parse_args()


if __name__ == "__main__":
    main(sys.argv[1:])
