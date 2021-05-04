#! /bin/bash

FOLDS=8

for m in sq101,20,128 \
         r18,20,128 ;do
 i=( ${m//,/ } )
 for f in $(seq 0 $((${FOLDS}-1)) );do
   python aqc_training.py --fold $f --folds $FOLDS --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]} 2>&1 |tee log_${i[0]}_${f}_${FOLDS}.txt
   python aqc_training.py --ref --fold $f --folds $FOLDS --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]}_ref --ref 2>&1 |tee log_ref_${i[0]}${f}_${FOLDS}.txt
 done
done
