# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018

import os
import collections

from skimage import io, transform

import torch
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader

QC_entry = collections.namedtuple( 
    'QC_entry',['id', 'status', 'qc_files', 'variant', 'cohort', 'subject', 'visit' ] )


def load_full_db(qc_db_path, data_prefix, validate_presence=False, feat=3, table="qc_all"):
    """Load complete QC database into memory
    """
    import sqlite3

    with sqlite3.connect(qc_db_path) as qc_db:
        query = f"select variant,cohort,subject,visit,path,xfm,pass from {table}"

        samples = []
        subjects = []

        for line in qc_db.execute(query):
            variant, cohort, subject, visit, path, xfm, _pass = line

            if _pass=='TRUE': status=1 
            else: status=0 

            _id='{}_{}_{}_{}'.format(variant, cohort, subject, visit)

            qc_files=[]
            for i in range(feat):
                qc_file='{}/{}/qc/aqc_{}_{}_{}.jpg'.format(data_prefix, path, subject, visit, i)
                
                if validate_presence and not os.path.exists(qc_file):
                    print("Check:", qc_file)
                else:
                    qc_files. append(qc_file)

            if len(qc_files)==feat:
                samples.append( QC_entry( _id, status, qc_files, variant, cohort, subject, visit ))
        
        return samples


def load_qc_images(imgs):
    ret = []
    for i, j in enumerate(imgs):
        try:
            im = io.imread(j)
        except :
            raise NameError(f"Problem reading {j}")
        assert im.shape == (224, 224)
        ret.append(torch.from_numpy(im).unsqueeze_(0).float()/255.0-0.5)
    return ret


def load_minc_images(path,winsorize_low=5,winsorize_high=95):
    from minc2_simple import minc2_file 
    import numpy as np

    input_minc=minc2_file(path)
    input_minc.setup_standard_order()

    sz=input_minc.shape
		
    input_images = [input_minc[sz[0]//2, :, :],
                    input_minc[:, :, sz[2]//2],
                    input_minc[:, sz[1]//2, :]]

     # normalize between 5 and 95th percentile
    _all_voxels=np.concatenate( tuple(( np.ravel(i) for i in input_images)) )
    # _all_voxels=input_minc[:,:,:] # this is slower
    _min=np.percentile(_all_voxels,winsorize_low)
    _max=np.percentile(_all_voxels,winsorize_high)
    input_images = [(i-_min)*(1.0/(_max-_min))-0.5 for i in input_images]
    
    # flip, resize and crop
    for i in range(3):
        # 
        _scale = min(256.0/input_images[i].shape[0],256.0/input_images[i].shape[1])
        # vertical flip and resize
        input_images[i] = transform.rescale(input_images[i][::-1, :], _scale, mode='constant', clip=False, anti_aliasing=False, multichannel=False)

        sz = input_images[i].shape
        # pad image 
        dummy = np.zeros((256, 256),)
        dummy[int((256-sz[0])/2): int((256-sz[0])/2)+sz[0], int((256-sz[1])/2): int((256-sz[1])/2)+sz[1]] = input_images[i]

        # crop
        input_images[i]=dummy[16:240,16:240]
   
    return [torch.from_numpy(i).float().unsqueeze_(0) for i in input_images]


def init_cv(dataset, fold=0, folds=8, validation=5, shuffle=False, seed=None):
    """
    Initialize Cross-Validation

    returns three indexes
    """
    n_samples   = len(dataset)
    whole_range = np.arange(n_samples)

    if shuffle:
        _state = None
        if seed is not None:
            _state = np.random.get_state()
            np.random.seed(seed)

        np.random.shuffle(whole_range)

        if seed is not None:
            np.random.set_state(_state)

    if folds > 0:
        training_samples = np.concatenate((whole_range[0:math.floor(fold * n_samples / folds)],
                                           whole_range[math.floor((fold + 1) * n_samples / folds):n_samples]))
        testing_samples = whole_range[math.floor(fold * n_samples / folds): math.floor((fold + 1) * n_samples / folds)]
    else:
        training_samples = whole_range
        testing_samples = whole_range[0:0]
    #
    validation_samples = training_samples[0:validation]
    training_samples = training_samples[validation:]

    return [dataset[i] for i in training_samples], \
           [dataset[i] for i in validation_samples], \
           [dataset[i] for i in testing_samples]

def split_dataset(all_samples, fold=0, folds=8, validation=5, 
    shuffle=False, seed=None, sec_samples=None):
    """
    Split samples, according to the subject field
    into testing,training and validation subsets
    sec_samples will be used for training subset, if provided
    """
    ### extract subject list
    subjects = set()
    for i in all_samples:
        subjects.add(i.subject)
    if sec_samples is not None:
        for i in sec_samples:
            subjects.add(i.subject)
    
    subjects=list(subjects)
    # split into three
    training_samples, validation_samples, testing_samples = init_cv(
        subjects, fold=fold,folds=folds, validation=validation, shuffle=shuffle,seed=seed
        )
    training_samples=set(training_samples)
    validation_samples=set(validation_samples)
    testing_samples=set(testing_samples)

    # apply index
    training = []
    validation = []
    testing = []

    for i in all_samples:
        if i.subject in testing_samples:
            testing.append(i)
        elif i.subject in validation_samples:
            validation.append(i)
    
    if sec_samples is not None:
        for i in sec_samples:
           if i.subject in training_samples:
               training.append(i)
    else:
        for i in all_samples:
           if i.subject in training_samples:
                training.append(i)
    
    return training, validation, testing


class QCDataset(Dataset):
    """
    QC images dataset. Uses sqlite3 database to load data
    """

    def __init__(self, dataset, data_prefix, use_ref=False):
        """
        Args:
            root_dir (string): Directory with all the data
            use_ref  (Boolean): use reference images
        """
        super(QCDataset, self).__init__()
        self.use_ref  = use_ref
        self.qc_samples = dataset
        self.data_prefix = data_prefix
        #
        self.qc_subjects = set(i.subject for i in self.qc_samples)

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                           [ self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                             self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                             self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg" ])

    def __len__(self):
        return len(self.qc_samples)

    def __getitem__(self, idx):
        _s = self.qc_samples[idx]
        # load images     
        _images = load_qc_images( _s.qc_files )

        if self.use_ref:
            _images = torch.cat( [ item for sublist in zip(_images, self.ref_img) for item in sublist ] )
        else:
            _images = torch.cat( _images )
        
        return {'image':_images, 'status':_s[1], 'id':_s.id}

    def n_subjects(self):
        return len(self.qc_subjects)

    def load_qc_db(self, data_prefix, feat=3, training_path=True):
        # load training list
        samples = []
        status = 2
        subjects = []
        
        # populate table with locations of QC jpg files
        if training_path:
            query = "select variant,cohort,subject,visit,path,xfm,pass from qc_all where subject not in (select subject from mem.val_subjects)"
        else:
            query = "select variant,cohort,subject,visit,path,xfm,pass from qc_all where subject in (select subject from mem.val_subjects)"

        for line in self.qc_db.execute(query):
            variant, cohort, subject, visit, path, xfm, _pass = line
            
            if _pass=='TRUE': 
                status=1 
            else: 
                status=0 
            
            _id = ':'.join((variant, cohort, subject, visit))
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


class MincVolumesDataset(Dataset):
    """
    Minc volumes dataset, loads slices from a list of images
    For inference in batch mode
    Arguments:
        file_list - list of minc files to load
        csv_file - name of csv file to load list from (first column)
    """
    def __init__(self, file_list=None, csv_file=None, winsorize_low=5, winsorize_high=95, 
                    use_ref=False, data_prefix=None):
        self.use_ref  = use_ref
        self.data_prefix = data_prefix
        self.winsorize_low=winsorize_low
        self.winsorize_high=winsorize_high

        if file_list is not None:
            self.file_list = file_list
        elif csv_file is not None:
            self.file_list=[]
            import csv
            for r in csv.reader(open(csv_file,'r')):
                self.file_list.append(r[0])
        else:
            self.file_list=[]

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                           [self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                            self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                            self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg"])
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        _images = load_minc_images(self.file_list[idx], winsorize_low=self.winsorize_low, winsorize_high=self.winsorize_high)

        if self.use_ref:
            _images = torch.cat( [ item for sublist in zip(_images, self.ref_img) for item in sublist ] )
        else:
            _images = torch.cat( _images )

        return _images.unsqueeze(0), self.file_list[idx]



class QCImagesDataset(Dataset):
    """
    QC images dataset, loads images identified by prefix in csv file
    Used for inference in batch mode only
    Arguments:
        file_list - list of QC images prefixes files to load
        csv_file - name of csv file to load list from (first column should contain prefix )
    """

    def __init__(self, file_list=None, csv_file=None, use_ref=False, data_prefix=None):
        self.use_ref  = use_ref
        self.data_prefix = data_prefix

        if file_list is not None:
            self.file_list = file_list
        elif csv_file is not None:
            self.file_list = []
            import csv
            for r in csv.reader(open(csv_file, 'r')):
                self.file_list.append(r[0])
        else:
            self.file_list = []

        if self.use_ref:
            # TODO: allow specify as parameter?
            self.ref_img = load_qc_images(
                           [self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                            self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                            self.data_prefix + os.sep + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg"])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        _images = load_qc_images( [ self.file_list[idx] + '_{}.jpg'.format(i) for i in range(3) ] )

        if self.use_ref:
            _images = torch.cat( [ item for sublist in zip(_images, self.ref_img) for item in sublist ] )
        else:
            _images = torch.cat( _images )
        return _images.unsqueeze(0), self.file_list[idx]
