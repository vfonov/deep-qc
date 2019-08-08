# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from datetime import datetime  # for tensorboard
import os
import sys
import math
import tensorflow as tf
# command line configuration
from tensorflow.python.platform import flags

# AQC models
from model import create_qc_model

# AQC estimator
from estimator import create_AQC_estimator,LoadEMAHook


# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be the name used when "
    "creating the Cloud TPU. To find out hte name of TPU, either use command "
    "'gcloud compute tpus list --zone=<zone-name>', or use "
    "'ctpu status --details' if you have created Cloud TPU using 'ctpu up'.")
# Model specific parameters
tf.flags.DEFINE_string(
    "model_dir", default="model",
    help="This should be the path of GCS bucket which will be used as "
    "model_directory to export the checkpoints during training.")
# Model specific parameters
tf.flags.DEFINE_string(
    "training_data", default="deep_qc_data_shuffled_20190805_train.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "testing_data", default="deep_qc_data_shuffled_20190805_test.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "validation_data", default="deep_qc_data_shuffled_20190805_val.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_integer(
    "batch_size", default=12,
    help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer(
    "eval_batch_size", default=8,
    help="This is the validation batch size.")
# flags.DEFINE_integer(
#     'num_cores', 1,
#     'Number of shards (workers).')
tf.flags.DEFINE_integer(
    "train_epochs", default=100,
    help="Total number of training epochs")
tf.flags.DEFINE_integer(
    "eval_per_epoch", default=1,
    help="Total number of training steps per evaluation")
# tf.flags.DEFINE_integer(
#     "eval_steps", default=4,
#     help="Total number of evaluation steps. If `0`, evaluation "
#     "after training is skipped.")
tf.flags.DEFINE_integer(
    "n_samples", default=57848,
    help="Number of samples")
tf.flags.DEFINE_integer(
    "n_val_samples", default=1097,
    help="Number of validation samples")
tf.flags.DEFINE_integer(
    "n_testing_samples", default=4246,
    help="Number of testing samples")
flags.DEFINE_float(
    'learning_rate', 1e-3, 'Initial learning rate')
tf.flags.DEFINE_integer(
    "learning_rate_decay_epochs", default=8, help="decay epochs")
flags.DEFINE_float(
    'learning_rate_decay', default=0.75, help="decay")
tf.flags.DEFINE_string(
    "optimizer", default="RMS",
    help="Training optimizer")
tf.flags.DEFINE_float(
    'depth_multiplier', default=1.0,
    help="mobilenet depth multiplier")
# tf.flags.DEFINE_bool(
#     "display_tensors", default=True,
#     help="display_tensors")

# TPU specific parameters.
tf.flags.DEFINE_bool(
    "use_tpu", default=False,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_bool(
    "moving_average", default=False,
    help="Use moving average")
# tf.flags.DEFINE_integer(
#     "iterations", default=500,
#     help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer(
    "save_checkpoints_secs", default=600,
    help="Saving checkpoint freq")
tf.flags.DEFINE_integer(
    "save_summary_steps", default=10,
    help="Saving summary steps")
tf.flags.DEFINE_bool(
    "log_device_placement", default=False,
    help="log_device_placement")

#MULTI-GPU specific paramters
tf.flags.DEFINE_bool(
    "multigpu", default=False,
    help="Use all available GPUs")

tf.flags.DEFINE_bool(
    "xla",default=False,
    help="Use xla compiler")

tf.flags.DEFINE_bool(
    "testing",default=False,
    help="Run in testing mode")
    

FLAGS = tf.flags.FLAGS

def load_data(batch_size=None, filenames=None, training=True):
    """
    Create training dataset
    """
    if batch_size is None:
        batch_size = FLAGS.batch_size

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    raw_ds = tf.data.TFRecordDataset(filenames)

    def _parse_feature(i):
        # QC data
        feature_description = {
            'img1_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'img2_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'img3_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qc':   tf.io.FixedLenFeature([], tf.int64,  default_value=0),
            'subj': tf.io.FixedLenFeature([], tf.int64,  default_value=0)
            #'_id':  tf.io.FixedLenFeature([], tf.string, default_value='')

        }
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(i, feature_description)

    def _decode_jpeg(a):
        img1 = tf.reshape(tf.cast(tf.image.decode_jpeg(
            a['img1_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0, [224,224,1] )
        img2 = tf.reshape(tf.cast(tf.image.decode_jpeg(
            a['img2_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0, [224,224,1] )
        img3 = tf.reshape(tf.cast(tf.image.decode_jpeg(
            a['img3_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0, [224,224,1] )
        # , 'subj':a['subj']
        return {'View1': img1, 'View2': img2, 'View3': img3}, {'qc': a['qc']}

    
    dataset = raw_ds.map(_parse_feature, num_parallel_calls=AUTOTUNE).map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    
    if training:
        # TODO: determine optimal buffer size, input should be already pre-shuffled
        #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=6000))
        dataset = dataset.shuffle(buffer_size=2000).repeat()

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def main(argv):
    del argv  # Unused
    #tf.logging.set_verbosity('WARN')

    if FLAGS.use_tpu:
        # validate input paths
        assert FLAGS.model_dir.startswith("gs://"), ("'model_dir' should be a "
                                                     "GCS bucket path!")

        assert FLAGS.validation_data.startswith("gs://"), ("'validation_data' should be a "
                                                     "GCS bucket path!")

        assert FLAGS.training_data.startswith("gs://"), ("'training_data' should be a "
                                                     "GCS bucket path!")
        
        _tpu = FLAGS.tpu if FLAGS.tpu is None else os.environ.get('TPU_NAME')
        # Resolve TPU cluster and runconfig for this.
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(_tpu)
    else:
        tpu_cluster_resolver = None

    #batch_size_per_shard = FLAGS.batch_size // FLAGS.num_cores
    #batch_axis = 0

    aqc_estimator = create_AQC_estimator(FLAGS, tpu_cluster_resolver=tpu_cluster_resolver)

    def _train_data(params): 
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=[FLAGS.training_data],
            training=True)
        return dataset

    def _eval_data(params):  
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=[FLAGS.validation_data],
            training=False)
        return dataset

    def _testing_data(params):  
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=[ FLAGS.testing_data ],
            training=False )
        return dataset

    if FLAGS.moving_average:
        eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
    else:
        eval_hooks = []

    steps_per_cycle = FLAGS.n_samples//FLAGS.batch_size//FLAGS.eval_per_epoch

    if not FLAGS.testing:
        for cycle in range(FLAGS.train_epochs * FLAGS.eval_per_epoch):
            #tf.logging.info('Starting training cycle %d.' % cycle)
            aqc_estimator.train(
                input_fn = _train_data,
                steps = steps_per_cycle)

            #tf.logging.info('Starting evaluation cycle %d .' % cycle)
            eval_results = aqc_estimator.evaluate(
                input_fn = _eval_data,
                hooks = eval_hooks,
                steps = math.ceil(FLAGS.n_val_samples/FLAGS.eval_batch_size)
                )
            tf.logging.info('Evaluation results: {}'.format(eval_results))
    else:
        aqc_estimator.warm_start_from(FLAGS.model_dir)
        # run evaluation on testing dataset
        testing_results = aqc_estimator.evaluate(
            input_fn = _testing_data,
            hooks = eval_hooks,
            steps = math.ceil(FLAGS.n_test_samples/FLAGS.eval_batch_size)
            )
        tf.logging.info('Testing results: {}'.format(testing_results))

            

if __name__ == '__main__':
    # main()
    tf.app.run()
    # tf.compat.v1.app.run()
