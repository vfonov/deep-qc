#! /usr/bin/env python
import os
import shutil
import scipy.linalg
import numpy as np
from minc2_simple import minc2_xfm

import sqlite3

# output directory
dir_pp='pp'
dir_model='avg.model'
dir_out='ref'

# model generation iteration to use
model_iter='001'

# reimplementation of xfmconcat using minc2_simple
def xfmconcat(inputs, output):
    # 
    all_linear=True
    input_xfms=[]
    input_grids=[]
    
    o=minc2_xfm()
    
    for j in inputs:
        try:
            x=minc2_xfm(j)
        except:
            print("Error reading:{} in {}".format(j,repr(inputs)))
            raise
        if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
            # this is a linear matrix
            o.append_linear_transform(x.get_linear_transform())
        else:
            raise Exception("Unexpected XFM type")

    o.save(output)


#avg.model/tal_ADNI_114_S_0416_m24_t1w.mnc_corr.001.xfm

dbi = sqlite3.connect('qc_db.sqlite3')

cur = dbi.cursor()
cur2 = dbi.cursor()
# create reference table
cur2.execute("CREATE TABLE IF NOT EXISTS qc_ref(cohort,subject,visit,ref_xfm)")
cur2.execute("delete from qc_ref")

cur.execute("select cohort,subject,visit,count(*) from qc_all where pass='TRUE' group by 1,2,3")

missing=0
present=0

for row in cur.fetchall():
    #print(name,group)
    (cohort,subject,visit,count)=row
    pp=os.path.join(dir_pp,cohort+'_'+subject,visit,'tal')
    # input files
    xfm_pp=os.path.join(pp,'tal_xfm_{}_{}_{}.xfm'.format(cohort,subject,visit))
    xfm_model=os.path.join(dir_model,'tal_{}_{}_{}_t1w.mnc_corr.{}.xfm'.format(cohort,subject,visit,model_iter))
    
    d=os.path.join(dir_out,cohort,subject,visit,'tal')
    xfm_out=os.path.join(d,'tal_xfm_{}_{}.xfm'.format(subject,visit))
    
    
    if not os.path.exists(xfm_pp):
      print('Missing:'+xfm_pp)
      missing+=1
    elif not os.path.exists(xfm_model):
      print('Missing:'+xfm_model)
      missing+=1
    else:
      if not os.path.exists(d):
        os.makedirs(d)
      xfmconcat([xfm_pp,xfm_model],xfm_out)
      cur2.execute("insert into qc_ref(cohort,subject,visit,ref_xfm) values (?,?,?,?)",(cohort,subject,visit,xfm_out))
      present+=1
      
print("Processed {} missing: {} present:{}".format(missing+present,missing,present))

dbi.commit()
# save reference csv
#ref_xfm.to_csv('ref_all.csv')
