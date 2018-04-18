#! /usr/bin/env python
import os
import shutil
import scipy.linalg
import numpy as np
from minc2_simple import minc2_xfm

import sqlite3

# output directory
outdir='avg'


def xfmavg(inputs, output):
    # TODO: handl inversion flag correctly
    all_linear=True
    all_nonlinear=True
    input_xfms=[]
    input_grids=[]
    
    for j in inputs:
        try:
            x=minc2_xfm(j)
        except:
            print("Error reading:{} in {}".format(j,repr(inputs)))
            raise
        if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
            # this is a linear matrix
            input_xfms.append(np.asmatrix(x.get_linear_transform()))
        else:
            raise Exception("Unexpected XFM type")
        
    acc=np.asmatrix(np.zeros([4,4],dtype=np.complex))
    for i in input_xfms:
        acc+=scipy.linalg.logm(i)
        
    acc/=len(input_xfms)
    acc=np.asarray(scipy.linalg.expm(acc).real,'float64','C')
    
    x=minc2_xfm()
    x.append_linear_transform(acc)
    x.save(output)


dbi = sqlite3.connect('qc_db.sqlite3')



cur = dbi.cursor()
cur2 = dbi.cursor()
# create reference table
cur2.execute("CREATE TABLE IF NOT EXISTS qc_ref(cohort,subject,visit,ref_xfm)")
cur2.execute("delete from qc_ref")

cur.execute("select cohort,subject,visit,count(*) from qc_all where pass='TRUE' group by 1,2,3")


for row in cur.fetchall():
    #print(name,group)
    (cohort,subject,visit,count)=row
    
    d=os.path.join(outdir,cohort,subject,visit,'tal')
    xfm_out=os.path.join(d,'tal_xfm_{}_{}.xfm'.format(subject,visit))
    
    if not os.path.exists(d):
        os.makedirs(d)
        #pass
    if count==1:
        # simple case , one reference
        cur2.execute("select xfm from qc_all where cohort=? and subject=? and visit=? and pass='TRUE'",(cohort,subject,visit))
        
        in_xfms=cur2.fetchone()[0]
        if not os.path.exists(xfm_out):
            print("Copy {} - {}".format(in_xfms,xfm_out))
            shutil.copyfile(in_xfms,xfm_out)
    else: 
        # need to average xfm files
        cur2.execute("select xfm from qc_all where cohort=? and subject=? and visit=? and pass='TRUE'",(cohort,subject,visit))
        in_xfms=[i[0] for i in cur2.fetchall()]
        if not os.path.exists(xfm_out):
            print("Average {} to {}".format(len(in_xfms),xfm_out))
            xfmavg(in_xfms,xfm_out)
        
    cur2.execute("insert into qc_ref(cohort,subject,visit,ref_xfm) values (?,?,?,?)",(cohort,subject,visit,xfm_out))
    #ref_xfm=ref_xfm.append({'cohort':cohort,'subject':subject,'visit':visit,'xfm':xfm_out},ignore_index=True)
    
dbi.commit()
# save reference csv
#ref_xfm.to_csv('ref_all.csv')
