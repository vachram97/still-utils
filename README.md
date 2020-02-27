## `analyzer` 

``` 
It is written in bash and uses only standard CrystFEL utils, such as 
process_hkl, partialator, compare_hkl, etc.

------
Usage:
------
	./analyzer -i output.stream \
		--merging process_hkl \
		--lowres 30.0 --highres 3.0 -y 222 --pushres 2.5

		Provides you with output.stream indexing stats 
		and performs merging with process_hkl
		Writes output to lysozyme.{hkl,hkl1,hkl2}.

	./analyzer -i lysozyme --rate \
		--highres 2.5 --lowres 20.0 --cell lyz_v2.cell

		Runs compare_hkl on previously generated hkl files
		with different unit cell file and resolution cutoffs

--------------------
Workflow parameters:
--------------------
	-i, --input
		Input file name (name or mask)
	--cell, -p
		Unit cell file (used for compare_hkl only)
	--merging
		How you want your merging to be done. 
		Options:
			none:	no merging will be performed
			process_hkl: merging with process_hkl will be performed
			partialator: merging with partialator will be done
	--rate
		You may want to rate previously processed HKL-file, but not perform any
		merging on stream. If that is true, input file name --input should be
		an HKL filename, rather than a stream. All three <>.[hkl, hkl1, hkl2]
		should be present.
	--multi_rmsd
		Used by multilattice_validate.py as cutoff (in degrees).
		If two lattices on the same image might be rotated on angle less than that,
		they will be considered as a false multiple.
	--nproc, -j
		Number of processors to be used for merging. By default, output of 'nproc'.
		If --nproc=0 or less, will use all availalbe and print warning.
	--logs
		A folder to write logs in. By default, logs/analyzer.<stream_md5sum>.<date>
	--histogram
		Whether to write histogram of resolutions or not (0 or 1).
		By default 1 -- does not build a histogram.
	--cleanup
		Whether you want to remove temporary files, such as concatenated stream,
		after the run is finished. Should be 0 or 1 (1 by default -- will clean up).
-------------------
Merging parameters:
-------------------
	--lowres, --highres
		Low and high resolution for output merging statistics table
	--pushres
		Merges reflections of higher than resolution_limit for each particular crystal,
		as predicted by peakfinder8 or peakfinder9.
		Quote from the process_hkl man:
		"Merge reflections which are up to n nm^-1 higher than the apparent resolution limit of each individual crystal.
		 n can be negative to merge lower than the apparent resolution limit."
		However, negative is yet disabled by this script, since it hasn't been useful so far.
	--iterations
		Number of iterations for scaling (used by partialator only)
	--highres_include
		Include in final merging only those crystals that have crystfel-predicted resolution higher than that.
		Used by process_hkl only (--min-res in man), partialator does not have this feature.
	--min_cc
		Quote from process_hkl man:
		 "Perform a second pass through the input, merging crystals only if their correlation with the initial model is at least n."
	-y, --symmetry
		Symmetry for merging. 
		Used by process_hkl/partialator and compare_hkl and check_hkl.
	--model
		Model for partialator. Should be either 'unity' or 'xsphere'.
	--string_params
		Parameters for either process_hkl or partialator provided as a string.
		Warning: it is your duty to make sure you provide valid parameters for
		the tool of your choice.
```

## `multilattice_validate.py` 

``` 
usage: multilattice_validate.py [-h] [--out OUT] [--rmsd RMSD] stream

Estimate number of falsely-multiple lattices

positional arguments:
  stream       Input stream

optional arguments:
  -h, --help   show this help message and exit
  --out OUT    Out list of multiple hits in a separate file
  --rmsd RMSD  RMSD (in deg) between 2 inverse lattices to be treated as
               separate ones
```

## `extract_cell.py` 

``` 
Usage: ./extract_cell.py my.stream
```

## `peaks2json.py` 

```
usage: peaks2json.py [-h] [--chunks CHUNKS] [--crystals CRYSTALS]
                     [--debug DEBUG]
                     stream

Indexed and located peak extraction from CrystFEL stream to json object

positional arguments:
  stream               input stream

optional arguments:
  -h, --help           show this help message and exit
  --chunks CHUNKS      Whether save chunk peaks or not
  --crystals CRYSTALS  Whether save crystal peaks or not
  --debug DEBUG        Don't supress lines with errors
```