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
out_filename = 'deep_qc_data'
out_subjects = 'deep_qc_subjects.csv'
feat=3

# make a query to the sqlite database that contains all the records
qc_db = sqlite3.connect(os.path.join(data_prefix, db_name))

qc_images=[] # QC images
qc_status=[] # QC status 1: pass 0: fail
qc_subject=[]  # subject id

for line in qc_db.execute("select variant,cohort,subject,visit,path,pass from qc_all"):
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

# hardcoded to work with three features

def serialize_dataset(images_jpeg, qc, subj ):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = { # TODO: fix this
    'img1_jpeg': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ images_jpeg[0].numpy() ] )),
    'img2_jpeg': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ images_jpeg[1].numpy() ] )),
    'img3_jpeg': tf.train.Feature( bytes_list = tf.train.BytesList(value=[ images_jpeg[2].numpy() ] )),
    'qc':   tf.train.Feature( int64_list = tf.train.Int64List(value=[ qc.numpy() ] )),
    'subj': tf.train.Feature( int64_list = tf.train.Int64List(value=[ subj.numpy() ] ))
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example( features=tf.train.Features(feature=feature) )
  return example_proto.SerializeToString()

def load_images(imgs,qc,subj):
    #a_ = tf.map_fn(tf.io.read_file, a)

    tf_string = tf.py_function(
        serialize_dataset, ( tf.map_fn(tf.io.read_file,imgs), qc, subj),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`
    return tf.reshape(tf_string, ()) # The result is a scalar

# pre-shuffle dataset
dataset_ds = dataset.map(load_images, num_parallel_calls=AUTOTUNE)

print("writing subject embedding to {}".format(out_subjects))
# save subject id embedding, just in case
with open(out_subjects,'w') as f:
  f.write("id,subject\n")
  for i,j in  qc_subjects_dict.items():
     f.write("{},{}\n".format(i,j))

#
shards=4

for s in range(shards):
  out_filename_='{}_{}.tfrecord'.format(out_filename,s)
  print("writing tfrecord to {}".format(out_filename_))
  ###########################
  writer = tf.data.experimental.TFRecordWriter(out_filename_)
  writer.write( dataset_ds.shard(shards,s))
