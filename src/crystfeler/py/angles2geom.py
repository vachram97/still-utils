#!/usr/bin/env python3


import argparse
import os
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R


def update_geom_params(
    initial_geom,
    alpha: np.float,
    beta: np.float,
    coffset: np.float,
    corner_x: np.float,
    corner_y: np.float,
    inplace=True,
) -> str:
    """
    Given an initial dict of values, 
    generates dict of geometry params, 
    given certain alpha, beta, coffset, corner_x and corner_y

    Arguments:
        initial_geom {[type]} -- initial geometry file
        alpha {np.float} -- angle of rotation along vertical axis (deg)
        beta {np.float} -- angle of rotation along horizontal axis (deg)
        coffset {np.float} -- crystal-to-detector length offset
        corner_x {np.float} -- x coordinate of detector corner
        corner_y {np.float} -- y coordinate of detector corner

    Returns:
        str -- updated filename
    """

    if inplace:
        geom_upd = initial_geom
    else:
        geom_upd = (
            f"{initial_geom[:-4]}_"
            "a_{alpha:.4f}__"
            "b_{beta:.4f}__"
            "corX_{corner_x:.2f}__"
            "corY_{corner_y:.2f}.geom"
        )

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

    answ = initial_dict.copy()
    answ[corner_x_key] = corner_x
    answ[corner_y_key] = corner_y
    answ[coffset_key] = coffset

    fs, ss = initial_dict[fs_key], initial_dict[ss_key]

    fs = f"{fs} + 0.0z" if "z" not in fs else fs
    ss = f"{ss} + 0.0z" if "z" not in ss else ss

    M = angles2matrix(fs, ss)
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])

    rotation_x = R.from_rotvec(alpha * x_axis)
    rotation_y = R.from_rotvec(beta * y_axis)
    M = rotation_y.apply(rotation_x.apply(M))

    coordinates = ("x", "y", "z")
    answ[fs_key] = " ".join(
        [f"{elem:+f}{name}" for elem, name in zip(M[0], coordinates)]
    )
    answ[ss_key] = " ".join(
        [f"{elem:+f}{name}" for elem, name in zip(M[1], coordinates)]
    )

    pr = []
    with open(initial_geom, "r") as fin:
        for idx, line in enumerate(fin):
            pr.append(line[:-1])
            for key in answ:
                if line and line[:-1].replace(" ", "").split("=")[0] == key:
                    pr = pr[:-1]
                    pr.append(" = ".join(map(str, (key, answ[key]))))
                    break

    with open(geom_upd, "w") as fout:
        print(*pr, sep="\n", file=fout)

    return geom_upd


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


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="""Applies angles to given geometry and saves updated geometry in a separate file"""
    )
    parser.add_argument("input_file", type=str, help="Input geometry file")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Alpha angle (rotation along x axis -- completes celling and beam to orthonormal)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="Beta angle (rotation along y axis -- along the beam)",
    )
    parser.add_argument("--coffset", type=float, default=0, help="corner_y value")
    parser.add_argument("--corner_x", type=float, help="corner_x value")
    parser.add_argument("--corner_y", type=float, help="corner_y value")

    args = parser.parse_args()

    ret = update_geom_params(
        args.input_file,
        alpha=args.alpha,
        beta=args.beta,
        corner_x=args.corner_x,
        corner_y=args.corner_y,
        coffset=args.coffset,
    )

    print(f"Saved geometry to {ret}")


if __name__ == "__main__":
    main(sys.argv[1:])
