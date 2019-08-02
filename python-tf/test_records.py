#! /usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()

rec='deep_qc_data_shuffled_20190801_train.tfrecord'


AUTOTUNE = tf.data.experimental.AUTOTUNE
raw_ds = tf.data.TFRecordDataset([rec])

def _parse_feature(i):
    feature_description = {
        'img1_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'img2_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'img3_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'qc':   tf.io.FixedLenFeature([], tf.int64,  default_value=0),
        'subj': tf.io.FixedLenFeature([], tf.int64,  default_value=0),
        '_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(i, feature_description)

def _decode_jpeg(a):
    img1 = tf.cast(tf.image.decode_jpeg(
        a['img1_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0
    img2 = tf.cast(tf.image.decode_jpeg(
        a['img2_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0
    img3 = tf.cast(tf.image.decode_jpeg(
        a['img3_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0

    # , 'subj':a['subj']
    return {'View1': img1, 'View2': img2, 'View3': img3}, {'qc': a['qc'],'id': a['_id']}

dataset = raw_ds.map(_parse_feature, num_parallel_calls=AUTOTUNE)
dataset = dataset.map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
#dataset = dataset.batch(10)
#summary_writer = tf.contrib.summary.create_file_writer('model/test', flush_millis=10000)

#with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
q=0
for a,b in dataset.take(100):
    #tf.contrib.summary.scalar("qc",   b['qc'], step=q)
    #pass_qc=tf.math.greater(b['qc'],0)
    #fail_qc=tf.math.less(b['qc'],1)

    img_qc=(tf.concat(
            [ a['View1'],
              a['View2'],
              a['View3']], axis=1 )+1)*127.5
    
    #print(q, pass)
    fname='{}_{}_{}.jpg'.format(q,b['qc'].numpy(),b['id'].numpy().decode("utf-8"))
    print(fname)
    tf.io.write_file(fname,tf.io.encode_jpeg(tf.cast(img_qc,np.uint8)))
    #tf.contrib.summary.image('Pass', img_pass_qc , max_images=1, step=q )
    #tf.contrib.summary.image('Fail', img_fail_qc , max_images=1, step=q )
    q+=1

