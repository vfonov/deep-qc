#! /bin/sh

data=../data

if false;then
# figure 3: network
dot aqc_net.dot -Tpng -Gsize=9,10\! -Gdpi=300  -o Figure_3.png
fi

# extracting samples:
#> select pass,cohort,count(*) from qc_all where variant='mritotal_std' group by 1,2;
#FALSE|ADNI|2402
#FALSE|HCP|69
#FALSE|PPMI_15T|52
#FALSE|PPMI_3T|204
#FALSE|PreventAD|268
#TRUE|ADNI|3770
#TRUE|HCP|828
#TRUE|PPMI_15T|170
#TRUE|PPMI_3T|566
#TRUE|PreventAD|971

#> select subject,visit,scan from qc_all where variant='mritotal_std' and cohort='ADNI' and pass="TRUE" limit 3;
#subject|visit|scan
#002_S_0295|bl|native/ADNI/002_s_0295_20060418_081744/002_s_0295_20060418_081744_4_mri.mnc
#002_S_0295|m06|native/ADNI/002_s_0295_20061102_081508/002_s_0295_20061102_081508_2_mri.mnc
#002_S_0295|m12|native/ADNI/002_s_0295_20070525_070639/002_s_0295_20070525_070639_3_mri.mnc

tempdir=$(mktemp -d --tmpdir)
trap "rm -rf $tempdir" 0 1 2 15

# passed 
for i in $(sqlite3 $data/qc_db.sqlite3 "select xfm from qc_all where variant='mritotal_std' and cohort='ADNI' and pass='TRUE'  ORDER BY RANDOM()  limit 3");do
  base=$(echo $i|cut -d / -f 1-4)
  nn=$(basename $i .xfm|cut -d _ -f 3-)
  echo $nn
  montage -font DejaVuSansMono -geometry +1+1 -tile 4x1 -gravity center -pointsize 30 -background white \
    -label '' label:Pass \
    -label '' $data/$base/qc/aqc_${nn}_0.jpg \
    -label '' $data/$base/qc/aqc_${nn}_1.jpg \
    -label '' $data/$base/qc/aqc_${nn}_2.jpg \
    $tempdir/pass_$nn.png
done

# fail
for i in $(sqlite3 $data/qc_db.sqlite3 "select xfm from qc_all where variant='mritotal_std' and cohort='ADNI' and pass='FALSE'  ORDER BY RANDOM()  limit 3");do
  base=$(echo $i|cut -d / -f 1-4)
  nn=$(basename $i .xfm|cut -d _ -f 3-)
  echo $nn
  montage -font DejaVuSansMono -geometry +1+1 -tile 4x1 -gravity center -pointsize 30 -background white \
    -label '' label:Fail \
    -label '' $data/$base/qc/aqc_${nn}_0.jpg \
    -label '' $data/$base/qc/aqc_${nn}_1.jpg \
    -label '' $data/$base/qc/aqc_${nn}_2.jpg \
    $tempdir/fail_$nn.png
done

# HACK 
for i in $tempdir/pass_*.png $tempdir/fail_*.png;do
  convert $i -gravity South -chop x40 $i
done

montage -geometry +1+1 -tile 1x6 \
  $tempdir/pass_*.png \
  $tempdir/fail_*.png \
  noref.png

rm -f $tempdir/pass_*.png $tempdir/fail_*.png

# with reference

# passed 
for i in $(sqlite3 $data/qc_db.sqlite3 "select xfm from qc_all where variant='mritotal_std' and cohort='ADNI' and pass='TRUE' ORDER BY RANDOM() limit 3 ");do
  base=$(echo $i|cut -d / -f 1-4)
  nn=$(basename $i .xfm|cut -d _ -f 3-)
  echo $nn
  montage -font DejaVuSansMono -geometry +1+1 -tile 7x1 -gravity center -pointsize 30 -background white \
    -label '' label:Pass \
    -label '' $data/$base/qc/aqc_${nn}_0.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg \
    -label '' $data/$base/qc/aqc_${nn}_1.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg \
    -label '' $data/$base/qc/aqc_${nn}_2.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg \
    $tempdir/pass_$nn.png
done


# fail
for i in $(sqlite3 $data/qc_db.sqlite3 "select xfm from qc_all where variant='mritotal_std' and cohort='ADNI' and pass='FALSE' ORDER BY RANDOM()  limit 3");do
  base=$(echo $i|cut -d / -f 1-4)
  nn=$(basename $i .xfm|cut -d _ -f 3-)
  echo $nn
  montage -font DejaVuSansMono -geometry +1+1 -tile 7x1 -gravity center -pointsize 30 -background white \
    -label '' label:Fail \
    -label '' $data/$base/qc/aqc_${nn}_0.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg \
    -label '' $data/$base/qc/aqc_${nn}_1.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg \
    -label '' $data/$base/qc/aqc_${nn}_2.jpg $data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg \
    $tempdir/fail_$nn.png
done

# HACK 
for i in $tempdir/pass_*.png $tempdir/fail_*.png;do
  convert $i -gravity South -chop x40 $i
done

montage -geometry +1+1 -tile 1x6 \
  $tempdir/pass_*.png \
  $tempdir/fail_*.png \
  ref.png



montage -geometry +20+1 -pointsize 30 -tile 2x1 \
  -label "A: Samples without reference" noref.png \
  -label "B: Samples with MNI ICBM 152 2009c reference" ref.png \
  Figure_2_rev1.png

exit

# figure 2: examples
montage -tile 4x3 \
  -geometry +1+1 \
    -pointsize 22 \
  -label "" \
  ../data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_0.jpg \
  ../data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_1.jpg \
  ../data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_2.jpg \
  -label "" label:"A: Passed QC" \
  ../data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_0.jpg \
  ../data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_1.jpg \
  ../data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_2.jpg \
  -label "" label:"B: Failed QC" \
  -label "Axial"    data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg \
  -label "Sagittal" data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg \
  -label "Coronal"  data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg \
  -label "" label:"C: Reference:\nMNI ICBM 152\n2009c" \
  Figure_2.png

