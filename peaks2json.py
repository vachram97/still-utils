#!/bin/sh

""":"
if [ -x "$(command -v pypy3)" ]; then
    echo "Running with pypy3"
    exec pypy3 $0 "$@" 
else
    echo "Running with python3. Consider using pypy (2x faster): http://pypy.org/download.html#installing"
    exec env python3 $0 "$@" 
fi
":"""

import os
import re
import json
import argparse
import sys
from tqdm import tqdm
import subprocess
from typing import List, Dict, Tuple


def parse_stream(filename: str, debug: bool) -> Tuple[Dict, Dict]:
    """
    Parses stream and returns all indexed and located by peakfinder8 peak positions

    Arguments:
        filename {str} -- input stream filename

    Returns:
        Dict -- dictionary containing all peak positions from crystals. Format is {('image_filename','event'):(panel,fs,ss)}
        Dict -- dictionary containing all peak positions from peakfinder. Format is {('image_filename','event'):(panel,fs,ss)}
    """

    stderr = sys.stderr if debug else open(os.devnull,'w')
    answ_crystals, answ_chunks = {}, {}

    def contains_filename(s):
        return s.startswith("Image filename")

    def contains_event(s):
        return s.startswith("Event")

    def contains_serial_number(s):
        return s.startswith("Image serial number")

    def starts_chunk_peaks(s):
        return s.startswith("  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel")

    def ends_chunk_peaks(s):
        return s.startswith("End of peak list")

    def starts_crystal_peaks(s):
        return s.startswith(
            "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"
        )

    def ends_crystal_peaks(s):
        return s.startswith("End of reflections")

    with open(filename, "r") as stream:
        is_chunk = False
        is_crystal = False
        current_filename = None
        current_event = None  # to handle non-event streams
        current_serial_number = None
        corrupted_chunk = False

        total_number_of_lines = subprocess.check_output(f'wc -l {filename}', shell=True)
        total_number_of_lines = int(total_number_of_lines.decode().split()[0])

        for line in tqdm(stream, desc="Reading stream", bar_format='{l_bar}{bar}{r_bar}', total=total_number_of_lines):
            try:
                if corrupted_chunk:
                    if 'Begin chunk' not in line:
                        continue
                    else:
                        is_crystal, is_chunk = False, False
                        corrupted_chunk = False
                        continue
                if contains_filename(line):
                    current_filename = line.split()[-1]
                elif contains_event(line):
                    current_event = line.split()[-1][2:]
                elif contains_serial_number(line):
                    current_serial_number = line.split()[-1]
                elif starts_chunk_peaks(line):
                    # image_id = (
                        # (current_filename, current_event, current_serial_number)
                        # if current_event is not None
                        # else (current_filename, current_serial_number)
                    # )
                    is_chunk = True
                    continue

                elif ends_chunk_peaks(line):
                    is_chunk = False
                    if current_event is not None:
                        answ_chunks[
                            (current_filename, current_event, current_serial_number)
                        ] = {"fs": float(fs), "ss": float(ss), "panel": panel}
                    else:
                        answ_chunks[(current_filename, current_serial_number)] = {
                            "fs": float(fs),
                            "ss": float(ss),
                            "panel": panel,
                        }

                elif starts_crystal_peaks(line):
                    is_crystal = True
                    continue
                elif ends_crystal_peaks(line):
                    is_crystal = False
                    if current_event is not None:
                        answ_crystals[
                            (current_filename, current_event, current_serial_number)
                        ] = {"fs": float(fs), "ss": float(ss), "panel": panel}
                    else:
                        answ_crystals[(current_filename, current_serial_number)] = {
                            "fs": float(fs),
                            "ss": float(ss),
                            "panel": panel,
                        }

            except Exception as e:
                print(f"    Caught {e} on current line: {line}", end='', file=stderr)
                corrupted_chunk = True
                continue

            # analyzing what we've got
            try:
                if is_chunk:
                    fs, ss, _, _, panel = [i for i in line.split()]
                elif is_crystal:
                    _, _, _, _, _, _, _, fs, ss, panel = [i for i in line.split()]
            except Exception as e:
                print(f"    Caught {e} on current line: {line}", end='', file=stderr)
                corrupted_chunk = True
                continue

    return answ_chunks, answ_crystals


def main(args: List[str]):
    """
    Main function
    
    Arguments:
        args {List[str]} -- input parameters
    """

    parser = argparse.ArgumentParser(
        description="Indexed and located peak extraction from CrystFEL stream to json object"
    )
    parser.add_argument("stream", help="input stream")
    parser.add_argument(
        "--chunks", help="Whether save chunk peaks or not", default=True
    )
    parser.add_argument(
        "--crystals", help="Whether save crystal peaks or not", default=True
    )
    parser.add_argument('--debug', help="Don't supress lines with errors", default=False)

    args = parser.parse_args()
    chunks, crystals = parse_stream(args.stream, debug=args.debug)

    if args.chunks:
        out_filename = f"{args.stream}_chunks.json"
        with open(out_filename, "w") as fout:
            json.dump(
                {"|".join(key): value for key, value in chunks.items()}, fout, indent=4
            )
            print(f"Wrote crystal peaks to {out_filename}")
    if args.crystals:
        out_filename = f"{args.stream}_crystals.json"
        with open(out_filename, "w") as fout:
            json.dump(
                {"|".join(key): value for key, value in crystals.items()},
                fout,
                indent=4,
            )
            print(f"Wrote crystal peaks to {out_filename}")


if __name__ == "__main__":
    main(sys.argv[1:])
