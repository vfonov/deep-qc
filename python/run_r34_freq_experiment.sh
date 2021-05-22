#! /bin/bash


for m in r34,30,180 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}_ref_long/final.pth ];then
 #python aqc_training.py --pretrained --freq 200 --save_final --save_best --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]}_ref_long 2>&1 |tee log_${i[0]}.txt
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
