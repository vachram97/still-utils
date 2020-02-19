#!/bin/bash

input="tmpstream"
output=merging_stats_$(md5sum $input | cut -c1-5).csv
dorate="-1"
symmetry='-1'
highres='3.0'
lowres='30.0'
iterations='0'
model='unity'
pushres="inf"
j="6"


while [[ $# -gt 1 ]]
do
key="$1"
cell_set=0
case $key in
    -i|--input)
    input="$2"
    shift # past argument
    ;;
    --dorate)
    dorate="$2"
    shift # past argument
    ;;
    -s|--symmetry)
    symmetry="$2"
    shift # past argument
    ;;
    --highres)
    highres="$2"
    shift # past argument
    ;;
    --lowres)
    lowres="$2"
    shift # past argument
    ;;
    --iterations)
    iterations="$2"
    shift # past argument
    ;;
    -m|--model)
    model="$2"
    shift # past argument
    ;;
    -p|--pushres)
    pushres="$2"
    shift # past argument
    ;;
    -h|--help)
    echo "Indexing analysis only:"
    echo "      ./indexing_analysis.sh output.stream"
    echo "Merging with process_hkl and analysis:"
    echo "      ./indexing_analysis.sh output.stream --dorate 0 --pushres 1.0 --highres 2.5 --lowres 30.0 --symmetry 222"
    echo "Merging with partialator and analysis:"
    echo "      ./indexing_analysis.sh output.stream --dorate 1 --pushres 1.0 --highres 2.5 --lowres 30.0 --symmetry 222"
    shift # past argument
    ;;
    -j|--nproc)
    j="$2"
    shift # past argument
    ;;
    -c|--cell)
    cell="$2"
    cell_set=1
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done


echo "----------===Begin log===----------"
echo "Streams for processing:"
for fle in "$@"; do
    if [ -f "$fle" ]
    then
        echo -"$fle" 2>&1 
        cat "$fle" >> tmpstream 2>&1
    fi
done


#----------------------------


# outputs to overall_stats.log statistics, obtained with check_hkl (SNR, multiplicity, N of refl, etc), and also Rsplit, CC and CC*.
function rate {
	rm stats[0-9].dat &>/dev/null
        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom rsplit --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >  stats1.dat
        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom cc     --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats2.dat
        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom ccstar --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats3.dat
        check_hkl tmp.hkl -y "$symmetry" -p "$cell"                                       --lowres="$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat > stats4.dat

        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom rsplit --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >  stats5.dat
        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom cc     --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats6.dat
        compare_hkl tmp.hkl1 tmp.hkl2 -y "$symmetry" -p "$cell" --fom ccstar --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat >> stats7.dat
        check_hkl tmp.hkl --nshells 1 -y "$symmetry" -p "$cell"                          --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >> stats8.dat
	paste stats4.dat <(awk '{print $3'} stats1.dat) <(awk '{print $3'} stats2.dat) <(awk '{print $3'} stats3.dat) | head -1        > overall_stats.csv
	paste stats4.dat  <(awk '{print $2}' stats1.dat)  <(awk '{print $2}' stats2.dat)  <(awk '{print $2}' stats3.dat) | tail -n +2 >> overall_stats.csv
	
	echo "   -------------------------------------------------------------------------------------------------------------------------------------------------------" >> overall_stats.csv
	paste stats8.dat  <(awk '{print $2}' stats5.dat)  <(awk '{print $2}' stats6.dat)  <(awk '{print $2}' stats7.dat) | tail -n +2 >> overall_stats.csv
}



echo "Filename for current run: $input"
echo "Stream generated by:  $(grep -a 'Generated by' "$input" | uniq)"

if [[ "$cell_set" -eq 1 ]] 
then 
	:
else
	cell=$(grep -a 'indexamajig' "$input" | awk '{for(i = 1; i <= NF;i++) if($i~/\-p/) print $(i+1)}' | tail -1 | uniq)
fi

pythonstring='from __future__ import print_function; print(*[i.split("-i")[1].split()[0] for i in open("'$input'").readlines() if "indexamajig" in i],sep="\n")'
NIMAGES_INPUT=$(python -c "$pythonstring" | xargs wc -l 2> /dev/null | tail -1 | awk '{print $1}')
if [[ "$NIMAGES_INPUT" -eq 0 ]]; then
	NIMAGES_INPUT="n/a (file lists not available)"
fi

#-----------------------

number_of_streams=$(grep -a 'indexamajig' $input | wc -l) # grep -as number of streams used for dorate processing
if [[ "$number_of_streams" -gt 1 ]]
then
	echo "Multi-stream mode; number of streams: $number_of_streams"	
	echo "indexamajig string: $(grep -a 'indexamajig' $input | tail -1)"
else
	echo "Single-stream mode; number of streams: 1"
	echo "indexamajig string: $(grep -a indexamajig $input)"
fi


echo "md5 checksum: $(md5sum $input)"
echo "Date: $(date -R)"

echo "================="
echo "Indexing details:"
echo "================="

NIMAGES=$(grep -a "Begin chunk" $input | wc -l )
NCRYST=$(grep -a "Begin crystal" $input | wc -l )

# lists all indexing methods used
METHODS=($(grep -E -a "indexed_by" "$input" | grep -a -v 'none' | sort | uniq | awk 'NF>1{print $NF}' | tr '\n' ' '))
NINDEXED=0

for i in "${METHODS[@]}"
do
	if [ "$i" = "none" ]
	then
		continue
	fi

	tmp="$(grep -E -c -a -w "$i" "$input")"
	let "NINDEXED=$NINDEXED+$tmp"
	ratio=$(echo " scale=3; $tmp/$NIMAGES" | bc)
	echo -e "$ratio" "\t" "$tmp" "\t" "$i"
done

NSPOTS=$(grep -a "num_reflections" "$input" | awk '{print $3;}' | paste -sd+ | bc)


echo "================="
echo "Indexing summary:"
echo "================="
echo "Total number of images for processing:	" $NIMAGES_INPUT
echo "Number of processed images:		" $NIMAGES
echo "Number of indexed:	" $NINDEXED
echo "Number of crystals:	" $NCRYST
echo "Number of spots found:	" $NSPOTS
#echo "Spots per image:	" $(echo "scale=2; $NSPOTS/$NIMAGES" | bc )
#echo "Spots per crystal:	" $(echo "scale=2; $NSPOTS/$NCRYST" | bc )
echo "Image indexing rate:		" $(echo "scale=2; $NINDEXED/$NIMAGES" | bc )
echo "Crystals percentage:	" $(echo "scale=2; $NCRYST/$NIMAGES" | bc)
echo "Average crystals per image:	" $(echo "scale=2; $NCRYST/$NINDEXED" | bc)


#echo "==================="
#echo "Resolution summary:"
#echo "==================="
#grep 'diffraction_resolution_limit' $input | awk '{print $6}' | sort -n > reslim.txt 
#python -c 'from text_histogram import histogram; histogram([float(elem) for elem in open("reslim.txt").read().split("\n") if elem and float(elem) < 10], buckets=15)'
#
#echo "======================="
#echo "Profile radius summary:"
#echo "======================="
#grep 'profile_radius' $input | awk '{print $3}' | sort -n > profile_radius.txt 
#python -c 'from text_histogram import histogram; histogram([float(elem) for elem in open("profile_radius.txt").read().split("\n") if elem], buckets=15)'


if [[ "$dorate" == "1" ]]; then
	# runs partialator to estimate rmeas and other foms
	partialator -i "$input" -o tmp.hkl --iterations "$iterations" -j "$j" --model "$model"  --push-res "$pushres" -y "$symmetry"  &> partialator.log
	rate
elif [[ "$dorate" == "0" ]]; then
    process_hkl -i "$input" -o tmp.hkl  -y "$symmetry" --min-res "$lowres" --push-res "$pushres"
    process_hkl -i "$input" -o tmp.hkl1 -y "$symmetry" --min-res "$lowres" --push-res "$pushres" --odd-only
    process_hkl -i "$input" -o tmp.hkl2 -y "$symmetry" --min-res "$lowres" --push-res "$pushres" --even-only
    rate
else
	:
fi	



if [[ "$dorate" == "-1" ]]; then
	# rate
	exit 0; fi

echo "================"
echo "Merging summary:"
echo "================"
echo "Merging stats backup file: $output"
echo "Input file: $input"
echo "Merging stats with partialator --iterations "$iterations" --model "$model" --lowres "$lowres" --highres "$highres" --push-res "$pushres":"


echo "================" >>  "$output"
echo "Merging summary:" >> "$output"
echo "================" >> "$output"
echo "Input file: $input"
echo "Merging stats with partialator --iterations "$iterations" --model "$model" --lowres "$lowres" --highres "$highres" --push-res "$pushres":" >> "$output"
cat overall_stats.csv >> "$output"


rm stats[0-9].dat
cat overall_stats.csv
