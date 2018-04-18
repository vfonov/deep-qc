#! /bin/sh

# figure 3: network
dot aqc_net.dot -Tpng -Gsize=9,15\! -Gdpi=300  -o results/paper_figure_3.png

# figure 2: examples
montage -tile 4x3 \
  -geometry +1+1 \
    -pointsize 22 \
  -label "" \
  data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_0.jpg \
  data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_1.jpg \
  data/bestlinreg_claude/ADNI/002_S_0295/m72/qc/aqc_002_S_0295_m72_2.jpg \
  -label "" label:"A: Passed QC" \
  data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_0.jpg \
  data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_1.jpg \
  data/bestlinreg-xcorr/ADNI/002_S_0619/m12/qc/aqc_002_S_0619_m12_2.jpg \
  -label "" label:"B: Failed QC" \
  -label "Axial"    mni_icbm152_t1_tal_nlin_sym_09c_0.jpg \
  -label "Sagittal" mni_icbm152_t1_tal_nlin_sym_09c_1.jpg \
  -label "Coronal"  mni_icbm152_t1_tal_nlin_sym_09c_2.jpg \
  -label "" label:"C: Reference:\nMNI ICBM 152\n2009c" \
      results/paper_figure_2.png

  
