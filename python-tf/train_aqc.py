# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals

import sqlite3
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.compat.v1.enable_eager_execution()
BATCH_SIZE = 128
steps_per_epoch = 200

#####
##### reader
##### 
#####

filename = 'deep_qc_data.tfrecord'
filenames = [ filename ]
raw_dataset = tf.data.TFRecordDataset( filenames )


feature_description = {
    'img1': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'img2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'img3': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'qc':   tf.io.FixedLenFeature([], tf.int64,  default_value=0 ),
    'subj': tf.io.FixedLenFeature([], tf.int64,  default_value=0 )
    }

def _parse_function(i):
    # Parse the input tf.Example proto using the dictionary above.
    a = tf.io.parse_single_example(i, feature_description)
    # print(tf.image.decode_jpeg(a['img1'],channels=1))
    a['img1'] = tf.image.decode_jpeg(a['img1'], channels=1)
    a['img2'] = tf.image.decode_jpeg(a['img2'], channels=1)
    a['img3'] = tf.image.decode_jpeg(a['img3'], channels=1)
    return  a['img1'], a['img2'], a['img3'], a['qc'], a['subj']


parsed_dataset = raw_dataset.map( _parse_function, num_parallel_calls=AUTOTUNE )
#parsed_dataset = parsed_dataset.repeat()
#parsed_dataset = parsed_dataset.batch( BATCH_SIZE )
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
#parsed_dataset = parsed_dataset.prefetch( buffer_size=AUTOTUNE )
#timeit(parsed_dataset)
#print(len(parsed_dataset))
c=0
for i in parsed_dataset.filter(lambda a,b,c,d,e:e>10):
    c+=1

print(c)
