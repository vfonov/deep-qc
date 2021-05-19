#! /bin/bash


for m in r18,20,112 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}_ref_long_l2_1e-4/final.pth ];then

 python aqc_training.py --pretrained --freq 200 --save_final --save_best \
  --folds 10 --fold 0 \
  --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]} \
  --lr 0.0001 --warmup_iter 100 --clip 1.0 --l2 1e-4 \
   model_${i[0]}_ref_long_l2_1e-4 2>&1 |tee log_${i[0]}.txt

 fi
done

