#! /bin/bash


for m in x50,20,12 \
         x101,20,6 \
         wr50,20,12 \
         wr101,24,6 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}/final.pth ];then
 python3 aqc_training.py --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]} 2>&1 |tee log_${i[0]}.txt

 fi

done
