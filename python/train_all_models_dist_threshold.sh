#! /bin/bash

lr=0.0001

threshold=15.80272 # based on LR

# train with reference
#         r34,10,128 \
#         r50,10,80 \
#         r101,10,48 \
#         r152,10,32 ;do

for m in \
         r18,10,196 \
        ;do
 i=( ${m//,/ } )
 for ref in Y N;do
    if [[ $ref == Y ]];then 
      suff='_ref'
      param='--ref'
    else
      suff=''
      param=''
    fi

    out=dist/model_thr_${i[0]}${suff}
    if [ ! -e $out/final.pth ];then
    mkdir -p $out
    python aqc_training.py \
        --lr $lr --warmup_iter 100 \
        --clip 1.0 --l2 0.0 \
        $param --pretrained \
        --adam \
        --fold 0 --folds 0 --net ${i[0]} \
        --n_epochs ${i[1]} --batch_size ${i[2]}  \
        --save_final --save_best --save_cpu --dist_threshold $threshold \
        $out 2>&1 |tee $out/log_${suff}_${i[0]}.txt
    fi
  done
done
