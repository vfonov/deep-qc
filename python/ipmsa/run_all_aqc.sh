#! /bin/bash

in=ipmsa_aqc_1000_list
out=ipmsa_aqc_1000_out

N=$(cat $in|wc -l )
echo "N=$N"

for model in r101 r152 r18 r34 r50 sq101 ;do

python ../aqc_apply.py \
 --raw --batch_pics --gpu --batch-size 8 --net $model --batch $in \
| tqdm --total=$N --desc=$model --ascii > aqc_raw_${model}

done

echo "Now join it all together!"


