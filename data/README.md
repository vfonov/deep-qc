# Miscellanious scripts for handling RAW data

`make_ref_xfm.py` - generate silver standard transformations, data is stored in sqlite database
`calc_difference2.py` - calculate distance metrics, data is stored in sqlite database

# RAW DATA (when it becomes public)

`bestlinreg_claude` - output of improved BestLinReg
`bestlinreg-mi` - output of the standard BestLinReg with mutual information
`bestlinreg-xcorr` -  output of the standard BestLinReg with cross-correlation
`elastix` - output of elastix
`mritotal_icbm` - output of mritotal in ICBM mode
`mritotal_std` - output of mritotal in standard mode
`avg`  - "silver" standard transformations, output of `make_ref_xfm.py`

# Torch data  (when it becomes public)
`models` - pre-trained models used as basis of deep-net
