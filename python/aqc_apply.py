#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018
import argparse
import os
import sys

import numpy as np
import io
import copy

from aqc_data import *
from model.util import *

import torch
import torch.nn as nn

from torch.autograd import Variable


default_data_dir=os.path.dirname(sys.argv[0])


def parse_options():

    parser = argparse.ArgumentParser(description='Apply automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ref",action="store_true",default=False,
                        help="Use reference images")
    parser.add_argument("--image", type=str, 
                        help="Input image prefix: <prefix>_{0,1,2}.jpg")
    parser.add_argument("--volume", type=str, 
                        help="Input minc volume (need minc2 simple)")
    parser.add_argument("--load", type=str, default=default_data_dir+'model_r18/best_tnr_cpu.pth',
                        help="Load pretrained model (mondatory)")
    parser.add_argument("--net", choices=['r18', 'r34', 'r50','r101','r152','sq101'],
                    help="Network type",default='r18')
    parser.add_argument('--raw', action="store_true", default=False,
                        help='Print raw score [0:1]')
    parser.add_argument('-q', '--quiet', action="store_true",default=False,   
                    help='Quiet mode, set status code to 0 - Pass, 1 - fail')

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    params = parse_options()
    use_ref = False

    if params.load is None:
        print("need to provide pre-trained model!")
        exit(1)
    
    if params.image is not None:
        inputs = load_qc_images([params.image+'_0.jpg',params.image+'_1.jpg',params.image+'_2.jpg'])
    elif params.volume is not None:
        inputs = load_minc_images(params.volume)
    else:
        print("Specify input volume or image prefix")
        exit(1)

    model = get_qc_model(params,use_ref=use_ref)
    model.train(False)

    # convert inputs into properly formated tensor
    # with a single batch dimension
    inputs = torch.cat( inputs ).unsqueeze_(0)
    inputs = Variable( inputs )

    softmax=nn.Softmax(dim=1)
    outputs = softmax.forward(model(inputs))
    _, preds = torch.max(outputs.data, 1)

    # raw score
    if params.raw:
        print(outputs.data[0,1])
    elif not params.quiet:
        if preds[0]==1:
            print("Pass")
        else:
            print("Fail")
    else:
        exit(0 if preds[0]==1 else 1)
