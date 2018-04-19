# /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018



from skimage import io, transform
import os
from minc2_simple import minc2_file 
import numpy as np


def minc_to_slices(path):

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
   
    return input_images


def parse_options():

    parser = argparse.ArgumentParser(description='Generate QC pics for deep_qc',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("volume", type=str, 
                        help="Input minc volume ")
    parser.add_argument("output", type=str, 
                        help="Output image prefix: <prefix>_{0,1,2}.jpg")

    params = parser.parse_args()
    
    return params


if __name__ == '__main__':
    params = parse_options()

    slices = minc_to_slices(params.volume)
    for _,i in enumerate(slices):
        io.imsave(params.output+"_{}.jpg".format(i),slices[i])
