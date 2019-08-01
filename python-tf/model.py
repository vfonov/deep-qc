# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals

#import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def augment_inner_model(inner_model, conv, x, out_filters=32, dropout=True):
    # pass through inner model to extract features
    x = inner_model(x)
    x = conv(x)
    # ResNetX style learning of high order features
    # TODO: replace with SeparableConv2D ?
    # x = layers.Conv2D(out_filters, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D( (3,3), activation='relu', padding='valid')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(out_filters, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    if dropout:
        x = layers.SpatialDropout2D(0.5)(x)

    return x

def create_qc_model(input_shape=(224, 224, 1), dropout=True, filters=32):

    # use existing create Keras model as base, reduce number of channels right away
    inner_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    conv = layers.Conv2D(filters, (1,1), activation='relu')
    #inner_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 1), include_top=False, weights=None)

    # create registration classification model
    im1 = layers.Input(shape=(224, 224, 1), name='View1')
    im2 = layers.Input(shape=(224, 224, 1), name='View2')
    im3 = layers.Input(shape=(224, 224, 1), name='View3')

    # use the same inner model for three images
    x1 = augment_inner_model(inner_model, conv, im1, dropout=dropout, out_filters=filters) 
    x2 = augment_inner_model(inner_model, conv, im2, dropout=dropout, out_filters=filters)
    x3 = augment_inner_model(inner_model, conv, im3, dropout=dropout, out_filters=filters)

    # join together
    x = layers.Concatenate(axis=-1)( [x1,x2,x3] )

    # learn spatial features of the merged layers
    x = layers.Conv2D(filters*3, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(16, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(4, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # output per region activation

    x = layers.Conv2D(1,(1,1))(x)
    x = layers.GlobalAveragePooling2D()(x) # average across all image

    # end of spatial preprocessing
    x = layers.Flatten(name='qc')(x)

    out_model = tf.keras.models.Model(inputs=[im1,im2,im3], outputs=x)

    return out_model
    
