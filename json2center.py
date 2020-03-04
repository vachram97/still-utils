#!/usr/bin/env python3

import numpy as np
import json
import sys
import argparse
from tqdm import tqdm
from typing import List


def circle(points: np.ndarray, acute_angle=True, permutations=True) -> np.ndarray:
    """
    Returns coordinates and radius of circumscribed circle for 3 points.
    
    Arguments:
        points {np.ndarray} -- (3,2)-shaped array
    
    Keyword Arguments:
        acute_angle {bool} -- whether the points should be an acute-anbled triangle (default: {True}) # TODO
        permutations {bool} -- whether to use all permutations for computation (default: {True}) # TODO
    
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
    answ = []

    for _ in tqdm(range(args.iterations), desc="Sampling points"):
        idx = np.random.randint(points.shape[0], size=3)
        T = points[idx]
        cur = circle(T)
        if cur is not None:
            answ.append(cur)

    xs, ys, rs = np.array(answ).T

    median_radius = np.median(rs)
    good_radii_mask = (rs < median_radius * 1.1) & (rs > median_radius * 0.9)

    print(f'Values: <x>: {xs[good_radii_mask].mean():.2f}\t<y>: {ys[good_radii_mask].mean():.2f}')
    print(f'Stds:   <x>: {xs[good_radii_mask].std():.2f}\t<y>: {ys[good_radii_mask].std():.2f}')

    # return answ


if __name__ == "__main__":
    main(sys.argv[1:])
