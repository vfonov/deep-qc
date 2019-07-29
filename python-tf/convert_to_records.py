# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019
from __future__ import absolute_import, division, print_function, unicode_literals

import sqlite3
import os
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.compat.v1.enable_eager_execution()

data_prefix='../data'
db_name='qc_db.sqlite3'
feat=3

# make a query to the sqlite database that contains all the records
qc_db = sqlite3.connect(os.path.join(data_prefix, db_name))
query = "select variant,cohort,subject,visit,path,pass from qc_all"

qc_images=[] # QC images
qc_status=[] # QC status 1: pass 0: fail
qc_subject=[]  # subject id

for line in qc_db.execute(query):
    variant, cohort, subject, visit, path, _pass = line

    if _pass=='TRUE': _status=1 
    else: _status=0 

    _id='%s_%s_%s_%s' % (variant, cohort, subject, visit)
    _subject='%s_%s' % (cohort,subject)

    _qc=[]

    for i in range(feat):
        qc_file = '{}/{}/qc/aqc_{}_{}_{}.jpg'.format(data_prefix, path, subject, visit, i)
        
        if not os.path.exists( qc_file ):
            print("Check:", qc_file)
        else:
            _qc.append( qc_file )

    if len( _qc ) == feat:
        qc_images.append(_qc)
        qc_status.append(_status)
        qc_subject.append(_subject)

# embed subject IDs
qc_subjects_dict = { j:i for i,j in enumerate(set(qc_subject)) }
qc_subject_idx = [ qc_subjects_dict[i] for i in qc_subject ]

dataset = tf.data.Dataset.from_tensor_slices( ( qc_images, qc_status, qc_subject_idx ) )

def serialize_dataset(img1, img2, img3, qc, subj ):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
    'img1': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ img1.numpy() ] )),
    'img2': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ img2.numpy() ] )),
    'img3': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ img3.numpy() ] )),
    'qc':   tf.train.Feature( int64_list = tf.train.Int64List(value=[ qc.numpy() ] )),
    'subj': tf.train.Feature( int64_list = tf.train.Int64List(value=[ subj.numpy() ] ))
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example( features=tf.train.Features(feature=feature) )
  return example_proto.SerializeToString()

def load_images(a,b,c):
    #a_ = tf.map_fn(tf.io.read_file, a)

    tf_string = tf.py_function(
        serialize_dataset,
         ( tf.io.read_file(a[0]), 
           tf.io.read_file(a[1]),
           tf.io.read_file(a[2]), b, c),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`

    return tf.reshape(tf_string, ()) # The result is a scalar

dataset_ds = dataset.map(load_images, num_parallel_calls=AUTOTUNE)

#
#
print("writing tfrecord")
###########################

filename = 'deep_qc_data.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset_ds)

