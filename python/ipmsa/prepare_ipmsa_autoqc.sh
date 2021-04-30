#! /bin/bash

in_pfx=/data_/ipmsa-preproc/Mahsa
out=aqc

mkdir -p $out
# V1,V2,jpg,QC
# scan,raw_score_old, manula_qc_jpg,manual_qc
# ../../Mahsa/Preproc/109MS301_subject_501-208/109MS303-w144/109MS301_subject_501-208_109MS303-w144_t1p_icbm.mnc

cat QC_Info_Balanced.csv|while read i;do 
j=(${i//,/ })

if [[ ${j[0]}  'V1']];then
scan=${in_pfx}/$(echo ${j[0]}|cut -d / -f 3-)

echo $scan ${j[1]} ${j[3]}
#if [[ -e out/$i/ses-2/aqc/aqc_${i}_ses-2_t1w_0.jpg ]];



fi
then