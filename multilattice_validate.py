#!/usr/bin/env python3

import numpy as np
import rmsd
from tqdm import tqdm
from typing import Dict, List


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
