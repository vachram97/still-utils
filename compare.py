#!/usr/bin/python3.6

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
from random import shuffle
from itertools import chain, product

f = h5py.File("1.h5", "r")
mat = f["/data/data"].value


def peaks(nparr):
    """Gives coordinates of values that are larger than all sidewise neighbours"""
    ret = np.zeros(nparr.shape).astype(np.int8)

    shifts_lst = product([-1, 1], repeat=2)
    shifts_lst = product([-1, 0, 1], repeat=2)
    shifts_lst = product([-2, -1, 0, 1, 2], repeat=2)
    shifts_lst = product(range(-3, 3 + 1), repeat=2)
    #  shifts_lst = product(range(-4, 4 + 1), repeat=2)

    for roll_combination in shifts_lst:
        ret += (np.roll(nparr, roll_combination, axis=(0, 1)) < nparr).astype(np.int8)

    return ret, np.nonzero(ret)


def save_peaks(nparr, pad=5, top=10, threshold=0.8):
    ret, (xs, ys) = peaks(nparr)
    if threshold < 1:  # normalizing by maximum smaller neighbours value
        threshold = int(ret.max() * threshold)

    cnt = 0
    for_iter = list(zip(xs, ys))
    shuffle(for_iter)
    for x, y in for_iter:
        if (
            (x - pad < 0)
            or (y - pad < 0)
            or (x + pad >= len(xs))
            or (y + pad >= len(ys))
            or (ret[x][y] < threshold)
        ):
            continue

        print(x, y)
        sns.heatmap(
            nparr[x - pad : x + pad, y - pad : y + pad],
            annot=True,
            fmt="d",
            cmap="Blues",
        )
        plt.savefig(f"peak_{x}_{y}.png", dpi=72)
        plt.clf()

        cnt += 1
        if cnt >= top:
            break
