#! /bin/bash

if true ;then
for m in r18  sq101  r34 r50 r101 r152 ;do
python3 aqc_training.py --net ${m} --adam --n_epochs 10 --batch_size 64  model_$m 2>&1 |tee log_${m}.txt

done
fi

if false;then
for m in r101 r152;do
python3 aqc_training.py --net ${m} --adam --n_epochs 10 --batch_size 16  model_$m 2>&1 |tee log_${m}.txt
done
fi

