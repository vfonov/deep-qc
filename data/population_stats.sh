#! /bin/bash

echo "Total number of unique scans"
sqlite3 qc_db.sqlite3 "select count(distinct scan) from qc_all;"
echo

echo "Total number of unique scans by cohort"
sqlite3 -column qc_db.sqlite3 "select cohort,count(distinct scan) from qc_all group by 1;"
echo


echo "Adni scans by field strength"
sqlite3 -column qc_db.sqlite3 "attach database 'adni_info.sqlite3' as ADNI; select cohort,FLDSTRENG,count(distinct scan) from qc_all left join ADNI.ADNIMERGE on ADNI.ADNIMERGE.PTID=qc_all.subject and ADNI.ADNIMERGE.VISCODE=qc_all.visit where qc_all.cohort='ADNI' group by 1,2;"
echo

echo "PASS/FAIL info"
sqlite3 -column qc_db.sqlite3 "select pass,count(*) from qc_all group by 1;"
echo

echo "augmented samples"
sqlite3 -column qc_db.sqlite3 "select count(*) from qc_all_aug ;"
echo


echo "Unique subjects"
sqlite3 -column qc_db.sqlite3 "select count(distinct subject) from qc_all ;"
echo
