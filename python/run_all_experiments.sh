#! /bin/bash


for m in sq101,20,64 \
         r18,20,64 \
         r34,30,32 \
         r50,30,16 \
         r101,30,12 \
         r152,40,8 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}/final.pth ];then
 python aqc_training.py --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]} 2>&1 |tee log_${i[0]}.txt
 fi

done

# run with reference
for m in sq101,20,64 \
         r18,20,64 \
         r34,30,32 \
         r50,30,16 \
         r101,30,12 \
         r152,40,8 ;do
 i=( ${m//,/ } )
 if [ ! -e model_${i[0]}_ref/final.pth ];then
 python aqc_training.py --net ${i[0]} --adam --n_epochs ${i[1]} --batch_size ${i[2]}  model_${i[0]}_ref --ref 2>&1 |tee log_${i[0]}.txt

 fi

done
