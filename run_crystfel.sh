#!/bin/bash

time=$(date "+%Y_%m_%d_%H_%M")

PROJECT_NAME="GPCR"
NPROC="32"

# PEAK FINDING PARAMETERS
SNR='3.0'
THRESHOLD='10'
HIGHRES='2.2'
SHUFFLE='1'


LST='YOUR_INPUT.lst' # your input list
shuf "$LST" > input.lst # your list must have events to enable this
if [[ "$SHUFFLE" == '1' ]]; then
	LST="input.lst";
else
	:
fi

GEOM="YOUR_GEOMETRY.geom"
CELL="YOUR_CELL.cell"

ln -s `pwd`/$(ls streams/* -trah | tail -n 1) laststream
indexamajig -i "$LST" \
-o "streams/${PROJECT_NAME}_${time}.stream" \
--profile \
-g "$GEOM" \
--peaks=peakfinder8 \
-j "$NPROC" \
-p "$CELL" \
--min-snr="$SNR" \
--threshold="$THRESHOLD" \
--highres="$HIGHRES" \
--indexing=mosflm,dirax,xds |& tee "log.indexamajig_${time}"
