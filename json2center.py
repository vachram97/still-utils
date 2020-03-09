#!/usr/bin/env python3

import numpy as np
import json
import sys
import argparse
from tqdm import tqdm
from typing import List
from itertools import combinations
import matplotlib.pyplot as plt


def radial_binning(fs: np.ndarray, ss: np.ndarray, rs: np.ndarray, N=1000) -> np.ndarray:
    """
    Returns binned by radius table

    Arguments:
        fs {np.ndarray} -- np.ndarray of x values
        ss {np.ndarray} -- np.ndarray of y values
        rs {np.ndarray} -- np.ndarray of r values

    Keyword Arguments:
        N {int} -- number of bins (default: {1000})

    Returns:
        answ {np.ndarray} -- (len(fs), 4) shape: rmean, num, fsmean, ssmean
    """
    rmin, rmax = rs.min(), rs.max()
    step = (rmax - rmin) / N
    answ = []
    for rcur, _ in enumerate(np.linspace(rmin, rmax, N)):
        mask = (rs < rcur + step) & (rs >= rcur)
        rmean = rs[mask].mean()
        num = mask.sum()
        fsmean = fs[mask].mean()
        ssmean = ss[mask].mean()
        answ.append([rmean, num, fsmean, ssmean])
    return np.array(answ)


def ang(v1, v2):
    """
    Returns angle between two vectors
    """
    return np.abs(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))


def is_inside(triangle: np.ndarray, point: np.ndarray) -> bool:
    """
    Checks whether point is inside a triangle

    Arguments:
        triangle {np.pdarray} -- triangle coordinates (3,2) shape
        point {np.ndarray} -- point to check (x,y)

    Returns:
        bool -- check value
    """

    a, b, c = triangle
    oa, ob, oc = a - point, b - point, c - point
    return (
        np.abs(
            sum([ang(v1, v2)
                 for v1, v2 in combinations([oa, ob, oc], 2)]) - 2 * np.pi
        )
        < 1e-2
    )


def circle(points: np.ndarray, acute_angle=True, presumable_centre=None) -> np.ndarray:
    """
    Returns coordinates and radius of circumscribed circle for 3 points.

    Arguments:
        points {np.ndarray} -- (3,2)-shaped array

    Keyword Arguments:
        acute_angle {bool} -- whether the points should be an acute-anbled triangle (default: {True})
        presumable_centre {np.ndarr} -- approximate centre position to check whether it's inside triangle of points

    Returns:
        np.ndarray -- (x,y,R)
    """
    x1, y1, x2, y2, x3, y3 = points.reshape(-1,)

    A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
    Y = np.array(
        [
            (x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2),
            (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2),
        ]
    )

    if acute_angle:
        if not is_inside(points, points.mean(axis=0)):
            return None

    if presumable_centre is not None:
        if not is_inside(points, presumable_centre):
            return None

    if np.abs(np.linalg.det(A)) < 1e-3:
        return None

    Ainv = np.linalg.inv(A)
    X = 0.5 * np.dot(Ainv, Y)
    x, y = X[0], X[1]
    r = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    return (x, y, r)


def main(args: List[str]):
    """
    The main function

    Arguments:
        args {List[str]} -- arguments
    """

    parser = argparse.ArgumentParser(
        description="Detector center search, based on random selection of circumscribed circle centers"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of times algorithm will try to guess center choosing 3 random points",
    )
    parser.add_argument(
        "input_json", type=str, help="input json file, produced by streampeaks2json.py",
    )

    args = parser.parse_args()

    with open(args.input_json) as f:
        peaks = json.load(f)
    number_of_panels = len(set([elem["panel"] for elem in peaks.values()]))
    assert number_of_panels == 1, f"Wrong number of panels: {number_of_panels}"

    points = np.array([(elem["fs"], elem["ss"]) for elem in peaks.values()])

    # presumable_centre = points.mean(axis=0)
    presumable_centre = np.array([719, 711])
    answ = []

    for _ in tqdm(range(args.iterations), desc="Sampling points"):
        idx = np.random.randint(points.shape[0], size=3)
        T = points[idx]
        cur = circle(T, acute_angle=True, presumable_centre=presumable_centre)
        if cur is not None:
            answ.append(cur)

    fs, ss, rs = np.array(answ).T

    return radial_binning(fs, ss, rs)


if __name__ == "__main__":
    # rmean, num, fsmean, ssmean = main(sys.argv[1:]).T
    points = main(sys.argv[1:])
