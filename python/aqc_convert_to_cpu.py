#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018
import argparse
import os
import numpy as np
#import h5py
import io
import copy

from aqc_data import *
from model.util import *

import torch
import torch.nn as nn

from torch.autograd import Variable


def parse_options():

    parser = argparse.ArgumentParser(description='Apply automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("load", type=str, default=None,
                        help="Load pretrained model (mondatory)")
    parser.add_argument("save", type=str, default=None,
                        help="Save pretrained model (mondatory)")
    parser.add_argument("--net", choices=['r18', 'r34', 'r50','r101','r152','sq101'],
                    help="Network type",default='r18')

    params = parser.parse_args()
    
    
    return params

if __name__ == '__main__':
    params = parse_options()
    use_ref = False
    #data_prefix="../data"
    if params.load is None or params.save is None:
        print("need to provide pre-trained model and output!")
        exit(1)
    
    model = get_qc_model(params,use_ref=use_ref)
    model.train(False)
    model = model.cpu()
    torch.save(model, params.save)




