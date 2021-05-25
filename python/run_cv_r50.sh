#! /bin/bash

FOLDS=8
lr=0.0001
pfx=lr_${lr}_pre
mkdir -p $pfx

# sq101,20,128
for m in r50,10,80 ;do
 i=( ${m//,/ } )
 for f in $(seq 0 $((${FOLDS}-1)) );do
   #python aqc_training.py --lr $lr --fold $f --pretrained --folds $FOLDS --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  $pfx/model_${i[0]} 2>&1 |tee $pfx/log_${i[0]}_${f}_${FOLDS}.txt
   #python aqc_training.py --lr $lr --ref --fold $f --pretrained --folds $FOLDS --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  $pfx/model_${i[0]}_ref --ref 2>&1 |tee $pfx/log_ref_${i[0]}${f}_${FOLDS}.txt
  python aqc_training.py \
      --lr $lr --warmup_iter 100 \
      --clip 1.0 \
      --l2 0.0 \
      --balance \
      --adam \
      --ref --fold $f --pretrained \
      --folds $FOLDS --net ${i[0]} \
      --n_epochs ${i[1]} --batch_size ${i[2]}  \
      $pfx/model_${i[0]}_ref --ref 2>&1 |tee $pfx/log_ref_${i[0]}${f}_${FOLDS}.txt
 done
done
