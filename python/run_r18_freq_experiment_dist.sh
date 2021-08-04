#! /bin/bash


for m in r18,30,196 ;do
 i=( ${m//,/ } )
 if [ ! -e model_dist_${i[0]}_ref_long/final.pth ];then

  python aqc_training.py \
      --save_final \
      --freq 200 \
      --lr 0.0001 --warmup_iter 100 \
      --clip 1.0 \
      --l2 0.0 \
      --adam \
      --ref --fold 0 --folds 8 --pretrained --dist \
      --net ${i[0]} \
      --n_epochs ${i[1]} --batch_size ${i[2]}  \
      model_dist_${i[0]}_ref_long --ref 2>&1 |tee log_dist_ref_${i[0]}_long.txt

 fi
done

