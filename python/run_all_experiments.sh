#! /bin/bash

if false;then
for m in sq101 r18;do
python3 aqc_training.py --net ${m} --adam --n_epochs 10 --batch_size 64  model_$m 2>&1 |tee log_${m}.txt
done
fi

if false ;then
for m in r34 ;do # sq101 r18  r34 r50 r101 r152 ;do
python3 aqc_training.py --net ${m} --adam --n_epochs 20 --batch_size 32  model_$m 2>&1 |tee log_${m}.txt

done
fi

if true;then
for m in  r50 r101 ;do
python3 aqc_training.py --net ${m} --adam --n_epochs 20 --batch_size 16  model_$m 2>&1 |tee log_${m}.txt
done
fi

if true;then
for m in  r152;do
python3 aqc_training.py --net ${m} --adam --n_epochs 30 --batch_size 8  model_$m 2>&1 |tee log_${m}.txt
done

fi

