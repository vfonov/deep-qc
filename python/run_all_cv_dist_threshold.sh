#! /bin/bash

FOLDS=8
lr=0.0001
threshold=15.80272 # based on LR

pfx=thr/lr_${lr}_pre
mkdir -p $pfx

for m in r18,10,196  ;do # r34,10,128 r50,10,80 r101,10,48 r152,10,32 
 i=( ${m//,/ } )
 for ref in Y N;do
   if [[ $ref == Y ]];then 
      suff='_ref'
      param='--ref'
   else
      suff=''
      param=''
   fi
   for f in $(seq 0 $((${FOLDS}-1)) );do
    if [[ ! -e $pfx/model_${i[0]}${suff}/log_${f}_${FOLDS}.json ]];then
        python aqc_training.py \
            --lr $lr --warmup_iter 100 \
            --clip 1.0 \
            --l2 0.0 \
            --adam \
            $param --fold $f --pretrained \
            --dist_threshold $threshold  \
            --folds $FOLDS --net ${i[0]} \
            --n_epochs ${i[1]} --batch_size ${i[2]}  \
            $pfx/model_${i[0]}${suff} 2>&1 |tee $pfx/log${suff}_${i[0]}_${f}_${FOLDS}.txt
    fi
   done
 done
done
