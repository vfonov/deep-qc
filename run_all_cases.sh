#! /bin/sh

#TODO: remove for publication
if false; then
# case 1 , long run to show early stopping benefit
th lua/aqc_training.lua -prefix results/r18_noref_long -run 1 -batches 10000 -r18

# run 8-fold cross-validation on all tested combinations:
th lua/aqc_training.lua -prefix results/r18_noref -run 8 -batches 1500 -r18
th lua/aqc_training.lua -prefix results/nin_noref -run 8 -batches 1500 

th lua/aqc_training.lua -prefix results/r18_ref -run 8 -batches 1500 -r18 -ref
th lua/aqc_training.lua -prefix results/nin_ref -run 8 -batches 1500 -ref

# experiment with reset, running for really long time, but one fold
th lua/aqc_training.lua -prefix results/r18_ref_reset -run 1 -batches 20000 -r18 -ref -reset

fi


# done



# dummy
th lua/aqc_training.lua -prefix results/dummy -run 1 -batches 100 -r18 -ref -reset
