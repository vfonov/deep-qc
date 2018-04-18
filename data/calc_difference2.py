#! /usr/bin/env python

import os
import shutil
import scipy.linalg
import numpy as np

from minc2_simple import minc2_xfm
import sqlite3


edges=[ [-60,-94, -52],
        [ 60, 50, 78] ]

def xfm_dist(xfm,ref):
    xfm1=minc2_xfm(xfm)
    xfm2=minc2_xfm(ref)
    #concatenate inverted xfm2
    xfm1.invert()
    xfm1.concat_xfm(xfm2)
    param=xfm1.get_linear_transform_param()
    
    if param.invalid:
      param.rotations.fill(float('nan'))
      
    param.dist=0.0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                p_in=np.array( [edges[x][0], edges[y][1], edges[z][2]] )
                p_out=xfm1.transform_point(p_in)
                dst=np.linalg.norm(p_in-p_out,ord=2)
                if dst>param.dist: param.dist=dst
    return param


# all input data
dbi = sqlite3.connect('qc_db.sqlite3')
cur = dbi.cursor()
cur2 = dbi.cursor()
# create reference table
cur2.execute("CREATE TABLE IF NOT EXISTS xfm_dist(variant,cohort,subject,visit,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz)")
cur2.execute("delete from xfm_dist")

# iterate over all 
cur.execute("select q.variant,q.cohort,q.subject,q.visit,q.xfm,r.ref_xfm from qc_all as q left join qc_ref as r on q.cohort=r.cohort and q.subject=r.subject and q.visit=r.visit")

for row in cur.fetchall():
    (variant,cohort,subject,visit,xfm,ref_xfm)=row
    
    d=xfm_dist(ref_xfm,xfm)
    
    cur2.execute("insert into xfm_dist(variant,cohort,subject,visit,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz) \
        values (:variant,:cohort,:subject,:visit,:lin,:rx,:ry,:rz,:tx,:ty,:tz,:sx,:sy,:sz)",
        {
            'variant':variant,
            'cohort':cohort,
            'subject':subject,
            'visit':visit,
            'lin':d.dist,
            'rx':d.rotations[0],
            'ry':d.rotations[1],
            'rz':d.rotations[2],
            'tx':d.translations[0],
            'ty':d.translations[1],
            'tz':d.translations[2],
            'sx':d.scales[0],
            'sy':d.scales[1],
            'sz':d.scales[2]
        }
        )
        
dbi.commit()

    
