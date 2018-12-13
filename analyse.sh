#!/bin/bash

time=$(date "+%Y_%m_%d_%H_%M")

PROJECT_NAME="GPCR"
NPROC="32"

# PEAK FINDING PARAMETERS
SNR='3.0'
THRESHOLD='10'
HIGHRES='2.2'


LST='imp_71-79.lst' # your input list
shuf "$LST" > input.lst # your list must have events to enable this

GEOM="yourgeom.geom"
CELL="yourcell.cell"


indexamajig -i input.lst \
-o "streams/"$PROJECT_NAME"_${time}.stream" \
--profile \
-g "$GEOM" \
--peaks=peakfinder8 \
-j "$NPROC" \
-p "$CELL" \
--min-snr="$SNR" \
--threshold="$THRESHOLD" \
--highres="$HIGHRES" \
--indexing=mosflm,dirax,xds |& tee log.indexamajig_$(time)

