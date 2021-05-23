#! /bin/bash


for m in r152,30,32 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}_ref_long/final.pth ];then
  python aqc_training.py \
      --freq 200 \
      --lr 0.0001 \
      --warmup_iter 100 \
      --clip 1.0 \
      --l2 0.0 \
      --balance \
      --adam \
      --ref --fold 0 --folds 8 --pretrained \
      --net ${i[0]} \
      --n_epochs ${i[1]} --batch_size ${i[2]}  \
      model_${i[0]}_ref_long --ref 2>&1 |tee log_ref_${i[0]}_ref_long.txt
 fi
done
