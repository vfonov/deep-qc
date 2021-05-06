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
    parser.add_argument("--net", choices=['r18', 'r34', 'r50','r101','r152',
                                          'sq101',
                                          'x50', 'x101',
                                          'wr50','wr101'],
                    help="Network type",default='r18')
    parser.add_argument("--ref",action="store_true",default=False,
                        help="Use reference images")

    params = parser.parse_args()
    
    
    return params

if __name__ == '__main__':
    params = parse_options()

    if params.load is None or params.save is None:
        print("need to provide pre-trained model and output!")
        exit(1)
    
    model = get_qc_model(params, use_ref=params.ref)
    model.train(False)
    model = model.cpu()

    print('Saving the model to {} ...' .format( params.save))
    torch.save(model.state_dict(), path)




