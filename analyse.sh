#!/bin/bash

time=$(date "+%Y_%m_%d_%H_%M")
echo streams/kr2_${time}.stream 

NPROC="35"
SNR='3.1'
THRESHOLD='10'
HIGHRES='2.2'
LST='imp_71-79.lst' # your input list
shuf "$LST" > input.lst

mv indexamajig.* logs_indexamajig

#
## indexing_analysis.sh c2_p/streams/c2_p_"$time".stream --slurm 0 --symmetry 2/m_uab  --highres 2.3 --push-res 1.5 > c2_p/streams/overall_stats_"$time".csv
#
## ls run* > c2_p/structure.txt
## cp *.* > c2_p
## rsync -avzP /xfel/ffhs/dat/ue_180124/marinegor/cheetah/c2_p/ greatthoughts@93.175.16.176:/media/greatthoughts/emarin_hdd_1TB/PAL_23-26.02.2018
## rsync -avP --ignore-existing run* c2_p/data-cheetah-2/
#
                                                                                                                                                             

