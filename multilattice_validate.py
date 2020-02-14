#!/usr/bin/env python3

import numpy as np
import rmsd
import sys
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Dict, List
import argparse


def read_stream(filename: str) -> Dict:
    """\
    Read stream and return dictionary {image_tag:[unit_cell_1, unit_cell_2,...]}
    """
    begin_chunk = lambda s: "Begin chunk" in s
    end_chunk = lambda s: "End chunk" in s
    begin_crystal = lambda s: "Begin crystal" in s
    end_crystal = lambda s: "End crystal" in s
    image_filename = lambda s: s.startswith("Image filename")
    event = lambda s: s.startswith("Event")
    has_inverse = lambda s: "star" in s and (
        s.startswith("astar") or s.startswith("bstar") or s.startswith("cstar")
    )
    has_cellparams = lambda s: s.startswith("Cell parameters")
    peaksfromlist = lambda s: "Peaks from peak search" in s

    def cell_params(line: str):
        """Return inverse lattice params extracted from the line"""
        return np.array([float(i) for i in line.split()[2:5]])

    active = False
    current_event = None  # to handle non-event streams
    current_chunk_crystals = []
    database = {}

    for line_num, line in tqdm(enumerate(open(filename)), "Reading stream"):
        if begin_chunk(line) or begin_crystal(line) or end_crystal(line):
            active = True
        if not active:
            continue
        else:
            if image_filename(line):
                current_filename = line.split()[-1]
            elif event(line):
                current_event = line.split()[-1]
            elif peaksfromlist(line):
                active = False
                continue
            elif has_cellparams(line):
                current_cell_params = cell_params(line)
            elif has_inverse(line):
                if line.startswith("astar"):
                    astar = cell_params(line)
                elif line.startswith("bstar"):
                    bstar = cell_params(line)
                elif line.startswith("cstar"):
                    cstar = cell_params(line)
                    active = False
                else:
                    raise TypeError(
                        f"Please check line {line}, num {line_num} -- it matches inverse param regex, but can not be parsed"
                    )
            elif end_crystal(line):
                current_chunk_crystals.append(
                    (np.array([astar, bstar, cstar]), current_cell_params)
                )
                continue
            elif end_chunk(line):
                active = False
                if len(current_chunk_crystals) > 0:
                    image_id = (
                        f"{current_filename} {current_event}"
                        if current_event is not None
                        else current_filename
                    )
                    database[image_id] = current_chunk_crystals
                current_chunk_crystals = []
                continue

    return database


def get_angle(vecs1, vecs2):
    """Returns rotation angle between two vector sets, representing inverse lattices"""

    # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
    return np.rad2deg(np.arccos((np.trace(rmsd.kabsch(vecs1, vecs2)) - 1) / 2))


def number_of_different_inverse_cells(vectors_set, rmsd_threshold=np.deg2rad(5.0)):
    """\
    Returns number of actually different cells in a given set,
    where cells that differ on rotation of less than `rmsd` radians are
    considered same
    """
    rmsd_matrix = [
        [1 if get_angle(v1, v2) < rmsd_threshold else 0 for v1 in vectors_set]
        for v2 in vectors_set
    ]
    graph = csr_matrix(rmsd_matrix)
    n_components = connected_components(
        csgraph=graph, directed=False, return_labels=False
    )
    return n_components


def main(args: List[str]):
    parser = argparse.ArgumentParser(
        description="Estimate number of falsely-multiple lattices"
    )
    parser.add_argument("stream", help="Input stream")
    parser.add_argument(
        "--rmsd",
        type=float,
        default=5.0,
        help="RMSD (in deg) between 2 inverse lattices to be treated as separate ones",
    )
    args = parser.parse_args()
    parsed_stream = read_stream(args.stream)
    parsed_lattices = [chunk[0] for chunk in parsed_stream if len(chunk) > 1]
    max_lattices_on_one_crystal = max([len(i) for i in parsed_lattices])

    for num_lattices in range(max_lattices_on_one_crystal):
        selected_lattices = [
            elem for elem in parsed_lattices if len(elem) == num_lattices
        ]
        components = [
            number_of_different_inverse_cells(elem) for elem in selected_lattices
        ]
        print(
            f"Crystals on image: {num_lattices}, total images: {len(selected_lattices)}, total lattices: {sum(components)}, false lattices: {len(selected_lattices)-sum(components)}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
