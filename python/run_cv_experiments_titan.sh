#! /bin/bash

FOLDS=8
lr=0.0001
#pfx=lr_${lr}_pre
pfx=titan
mkdir -p $pfx

for m in r18,4,98 ;do
 i=( ${m//,/ } )
 for f in $(seq 0 $((${FOLDS}-1)) );do
  CUDA_VISIBLE_DEVICES=1 python aqc_training.py \
    --lr $lr --warmup_iter 100 --clip 1.0 --l2 1e-6 \
    --adam \
    --ref --fold $f --pretrained \
    --folds $FOLDS --net ${i[0]} \
    --n_epochs ${i[1]} --batch_size ${i[2]}  \
     $pfx/model_${i[0]}_ref --ref 2>&1 |tee $pfx/log_ref_${i[0]}${f}_${FOLDS}.txt
 done
done
