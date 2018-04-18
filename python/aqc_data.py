# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018

import sqlite3
import os
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader

def load_qc_images(imgs):
    ret=[]  
    for i,j in enumerate(imgs):
        im=io.imread(j)
        assert im.shape==(224,224)
        ret.append(torch.from_numpy(im).unsqueeze_(0).float()/255.0-0.5)
    return ret

def load_minc_images(path):
    from minc2_simple import minc2_file 
    import numpy as np

    input_minc=minc2_file(path)
    input_minc.setup_standard_order()

    #load into standard volume
    sample=input_minc.load_complete_volume(minc2_file.MINC2_FLOAT)
    
    # normalize input
    _min=np.min(sample)
    _max=np.max(sample)
    sample=(sample-_min)*(1.0/(_max-_min))-0.5

    sz=sample.shape
    
    input_images=[sample[int(sz[0]/2),:,:],
                  sample[:,:,int(sz[2]/2)],
                  sample[:,int(sz[1]/2),:] ]
    
    # flip, resize and crop
    for i in range(3):
        # 
        _scale=min(256.0/input_images[i].shape[0],256.0/input_images[i].shape[1])
        # vertical flip and resize
        input_images[i]=transform.rescale(input_images[i][::-1,:], _scale, mode='constant', clip=False)

        sz=input_images[i].shape
        # pad image 
        dummy=np.zeros((256,256),)
        dummy[int((256-sz[0])/2) : int((256-sz[0])/2)+sz[0], int((256-sz[1])/2): int((256-sz[1])/2)+sz[1]] = input_images[i]

        # crop
        input_images[i]=dummy[16:240,16:240]
   
    # io.imsave("debug_0.png",input_images[0])
    # io.imsave("debug_1.png",input_images[1])
    # io.imsave("debug_2.png",input_images[2])
    return [ torch.from_numpy(i).float().unsqueeze_(0) for i in input_images]


class QCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, db, data_prefix, use_ref=False, validate=False, training_path=True):
        """
        Args:
            root_dir (string): Directory with all the data
            use_ref  (Boolean): use reference images
        """
        super(QCDataset, self).__init__()
        self.use_ref  = use_ref
        self.validate = validate
        self.training = training_path
        self.data_prefix = data_prefix
        #
        self.qc_db=db
        self.qc_samples,self.qc_subjects=self.load_qc_db(self.data_prefix, training_path=training_path)

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img=load_qc_images([self.root_dir+os.sep+"/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                            self.root_dir+os.sep+"/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                            self.root_dir+os.sep+"/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg"])

    def __len__(self):
        return len(self.qc_samples)

    def __getitem__(self, idx):
        
        _s=self.qc_samples[idx]
        # load images     
        _images = load_qc_images(_s[2] )

        if self.use_ref:
            _images = torch.cat( [ item for sublist in zip(_images,self.ref_img) for item in sublist ] )
        else:
            _images = torch.cat( _images )
            
        return {'image':_images, 'status':_s[1], 'id':_s[0]}

    def n_subjects(self):
        return len(self.qc_subjects)

    def load_qc_db(self, data_prefix, feat=3,training_path=True):
        # load training list
        samples=[]
        status=2
        subjects=[]
        
        # populate table with locations of QC jpg files
        if training_path:
            query="select variant,cohort,subject,visit,path,xfm,pass from qc_all where subject not in (select subject from mem.val_subjects)" 
        else:
            query="select variant,cohort,subject,visit,path,xfm,pass from qc_all where subject in (select subject from mem.val_subjects)" 

        for line in self.qc_db.execute(query):
            variant,cohort,subject,visit,path,xfm,_pass=line
            
            if _pass=='TRUE': status=1 
            else: status=0 
            
            _id='%s_%s_%s_%s' % (variant, cohort, subject, visit)
            qc=[]
            
            for i in range(feat):
                qc_file='{}/{}/qc/aqc_{}_{}_{}.jpg'.format(data_prefix, path, subject, visit, i)
                

                if self.validate and not os.path.exists(qc_file):
                    print("Check:",qc_file)
                else:
                    qc.append(qc_file)

            if len(qc)==feat:
                samples.append([ _id, status, qc, variant, cohort, subject, visit ])
        
        if training_path:
            query="select subject from all_subjects where subject not in (select subject from mem.val_subjects)" 
        else:
            query="select subject from mem.val_subjects" 

        # make a list of all subjects
        for line in self.qc_db.execute( query ) :
            subjects.append(line[0])
        
        return samples,subjects

