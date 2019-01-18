#! /bin/bash


for m in sq101,10,64 \
         r18,10,64 \
         r34,20,32 \
         r50,20,16 \
         r101,24,12 \
         r152,30,8 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}/final.pth ];then
 python3 aqc_training.py --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]} 2>&1 |tee log_${i[0]}.txt

 fi

done
