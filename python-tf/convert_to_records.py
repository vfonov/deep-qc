# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sqlite3
import os
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.compat.v1.enable_eager_execution()

def parse_options():

    parser = argparse.ArgumentParser(description='Convert QC data into tfrecords',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data", default="../data",
                        help="Data prefix")
    parser.add_argument("--shard", type=int, default=0,
                        help="Split data into pieces")
    parser.add_argument("--limit", type=int, default=0,
                        help="make a limited subset")
    parser.add_argument("--training", type=int, default=90,
                        help="training fraction")
    parser.add_argument("--testing", type=int, default=8,
                        help="testing fraction")
    parser.add_argument("--validation", type=int, default=2,
                        help="validation fraction")
    parser.add_argument("--shuffle",action="store_true",default=False,
                        help="Shuffle input")
    parser.add_argument("output", type=str, 
                        help="Output prefix")

    params = parser.parse_args()
    return params

if __name__ == '__main__':
    params = parse_options()

    data_prefix = params.data
    db_name = 'qc_db.sqlite3'
    out_filename = params.output
    out_subjects = out_filename + '_subjects.csv'

    # consf for now
    feat = 3

    # make a query to the sqlite database that contains all the records
    qc_db = sqlite3.connect(os.path.join(data_prefix, db_name))

    qc_images  = [] # QC images
    qc_status  = [] # QC status 1: pass 0: fail
    qc_subject = [] # subject id

    query = "select variant,cohort,subject,visit,path,pass from qc_all"
    if params.shuffle:
        query += " order by random()"
    if params.limit>0:
        query += " limit {}".format( params.limit )
    for line in qc_db.execute(query):
        variant, cohort, subject, visit, path, _pass = line

        _status=1 if _pass == 'TRUE' else 0

        _id = '%s_%s_%s_%s' % (variant, cohort, subject, visit)
        _subject =  '%s_%s' % (cohort,subject)

        _qc = []

        for i in range(feat):
            qc_file = '{}/{}/qc/aqc_{}_{}_{}.jpg'.format(data_prefix, path, subject, visit, i)
            if not os.path.exists( qc_file ):
                print("Check:", qc_file)
            else:
                _qc.append( qc_file )

        if len( _qc ) == feat:
            qc_images.append( _qc )
            qc_status.append( _status )
            qc_subject.append( _subject )

    # embed subject IDs
    qc_subjects_dict = { j:i for i,j in enumerate(set(qc_subject)) }
    qc_subject_idx = [ qc_subjects_dict[i] for i in qc_subject ]

    dataset = tf.data.Dataset.from_tensor_slices( ( qc_images, qc_status, qc_subject_idx ) )

    np.random.seed(42) # specify random seed, so that split is consistent
    # initialize subject-based split
    n_subj = len(qc_subjects_dict)
    all_subjects = np.random.permutation(n_subj)

    train_subjects = tf.convert_to_tensor(all_subjects[0:n_subj*params.training//100],dtype=np.int32 )
    testing_subjects = tf.convert_to_tensor(all_subjects[n_subj*params.training//100:n_subj*params.training//100+n_subj*params.testing//100] ,dtype=np.int32)
    validation_subjects = tf.convert_to_tensor(all_subjects[n_subj*params.training//100+n_subj*params.testing//100:-1] ,dtype=np.int32)

    all_sets = { 'train':train_subjects, 
                 'val':validation_subjects, 
                 'test':testing_subjects }

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
        'qc':        tf.train.Feature( int64_list = tf.train.Int64List(value=[ qc.numpy() ] )),
        'subj':      tf.train.Feature( int64_list = tf.train.Int64List(value=[ subj.numpy() ] ))
      }
      # Create a Features message using tf.train.Example.
      example_proto = tf.train.Example( features=tf.train.Features(feature=feature) )
      return example_proto.SerializeToString()

    def load_images(imgs, qc, subj):
        tf_string = tf.py_function(
            serialize_dataset, ( tf.map_fn(tf.io.read_file,imgs), qc, subj),  # pass these args to the above function.
            tf.string)      # the return type is `tf.string`
        return tf.reshape(tf_string, ()) # The result is a scalar


    #print("writing subject embedding to {}".format( out_subjects))
    # save subject id embedding, just in case
    # with open(out_subjects,'w') as f:
    #   f.write("id,subject\n")
    #   for i,j in  qc_subjects_dict.items():
    #     f.write("{},{}\n".format(i,j))

    for l,s in all_sets.items():
        dataset_ds = dataset.\
            filter(lambda im,qc,subj:tf.reduce_any( tf.math.equal(tf.expand_dims(subj, 0), tf.expand_dims(s,1)) )).\
            map( load_images, num_parallel_calls=AUTOTUNE)
        out_filename_ = '{}_{}.tfrecord'.format(out_filename,l)
        print("writing training tfrecord to {}".format(out_filename_))
        ###########################
        writer = tf.data.experimental.TFRecordWriter(out_filename_)
        writer.write( dataset_ds )
