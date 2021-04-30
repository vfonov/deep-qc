#! /bin/bash

in_pfx=/data_/ipmsa-preproc/Mahsa
out=aqc

mkdir -p $out
# V1,V2,jpg,QC
# scan,raw_score_old, manula_qc_jpg,manual_qc
# ../../Mahsa/Preproc/109MS301_subject_501-208/109MS303-w144/109MS301_subject_501-208_109MS303-w144_t1p_icbm.mnc

echo id,scan,aqc_base,old_score,manual_qc > ipmsa_aqc_1000.csv
rm -f ipmsa_aqc_1000_list

cat QC_Info_Balanced.csv|while read i;do 
j=(${i//,/ })

if [[ ${j[0]} != 'V1' ]]; then

id=$(echo ${j[0]}|cut -d / -f 7|sed -e 's/.mnc$//')
scan=${in_pfx}/$(echo ${j[0]}|cut -d / -f 4-)
out_file=${out}/$(echo ${j[0]}|cut -d / -f 5-|sed -e 's/.mnc$//')
out_dir=$(dirname $out_file)
mkdir -p $out_dir
echo $id,$scan,$out_file,${j[1]},${j[3]} >> ipmsa_aqc_1000.csv
echo $out_file >> ipmsa_aqc_1000_list
echo minc_aqc.pl $scan $out_file

#if [[ -e out/$i/ses-2/aqc/aqc_${i}_ses-2_t1w_0.jpg ]];
fi
done | parallel -j 4 --eta

echo "Now it's time to run DARQ"



