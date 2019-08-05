# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from datetime import datetime  # for tensorboard
import os
import sys

import tensorflow as tf
# command line configuration
from tensorflow.python.platform import flags
# TPU enabled models from  https://github.com/tensorflow/tpu/
# add local copy of tpu module:
from os.path import dirname
sys.path.append(os.path.join(dirname(__file__),'tpu/models/'))

import official.mobilenet.mobilenet_model as mobilenet_v1

# local
# from model import create_qc_model

# from tensorflow.contrib.framework.python.ops import arg_scope
# from tensorflow.contrib.training.python.training import evaluation

# slim tensorflow library
slim = tf.contrib.slim

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
    "training_data", default="deep_qc_data_shuffled_20190801_train.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "testing_data", default="deep_qc_data_shuffled_20190801_test.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "validation_data", default="deep_qc_data_shuffled_20190801_val.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_integer(
    "batch_size", default=16,
    help="This is the global batch size and not the per-shard batch.")
flags.DEFINE_integer(
    'num_cores', 1,
    'Number of shards (workers).')
tf.flags.DEFINE_integer(
    "train_epochs", default=100,
    help="Total number of training epochs")
tf.flags.DEFINE_integer(
    "eval_per_epoch", default=3,
    help="Total number of training steps per evaluation")
# tf.flags.DEFINE_integer(
#     "eval_steps", default=4,
#     help="Total number of evaluation steps. If `0`, evaluation "
#     "after training is skipped.")
tf.flags.DEFINE_integer(
    "n_samples", default=57848,
    help="Number of samples")
flags.DEFINE_float(
    'learning_rate', 1e-3, 'Initial learning rate')
tf.flags.DEFINE_integer(
    "learning_rate_decay_epochs", default=4, help="decay epochs")
flags.DEFINE_float(
    'learning_rate_decay', default=0.75, help="decay")
tf.flags.DEFINE_string(
    "optimizer", default="RMS",
    help="Training optimizer")
tf.flags.DEFINE_float(
    'depth_multiplier', default=1.0,
    help="mobilenet depth multiplier")
tf.flags.DEFINE_bool(
    "display_tensors", default=True,
    help="display_tensors")
# TPU specific parameters.
tf.flags.DEFINE_bool(
    "use_tpu", default=False,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_integer(
    "iterations", default=500,
    help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer(
    "save_checkpoints_secs", default=600,
    help="Saving checkpoint freq")
tf.flags.DEFINE_integer(
    "save_summary_steps", default=10,
    help="Saving summary steps")
tf.flags.DEFINE_bool(
    "log_device_placement", default=False,
    help="log_device_placement")

FLAGS = tf.flags.FLAGS

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


def create_inner_model(images, scope=None,is_training=True,reuse=False):
    with tf.variable_scope(scope,reuse=reuse) as _scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                net, _ = mobilenet_v1.mobilenet_v1_base(images, scope=_scope)
                net = slim.conv2d(net, 1024, [1, 1])
                net = slim.conv2d(net, 512,  [1, 1])
    return net

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
        return {'View1': img1, 'View2': img2, 'View3': img3}, {'qc': a['qc']}

    
    dataset = raw_ds.map(_parse_feature, num_parallel_calls=AUTOTUNE).map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    
    if training:
        # TODO: determine optimal buffer size, input should be already pre-shuffled
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=6000))
        # dataset = dataset.shuffle(buffer_size=2000)
        # dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def model_fn(features, labels, mode, params):
    """Mobilenet v1 model using Estimator API."""
    num_classes = 2
    batch_size = params['batch_size']

    training_active = (mode == tf.estimator.ModeKeys.TRAIN)
    eval_active = (mode == tf.estimator.ModeKeys.EVAL)
    predict_active = (mode == tf.estimator.ModeKeys.PREDICT)

    images = features

    images1 = tf.reshape(images['View1'], [batch_size, 224, 224, 1])
    images2 = tf.reshape(images['View2'], [batch_size, 224, 224, 1])
    images3 = tf.reshape(images['View3'], [batch_size, 224, 224, 1])
    labels  = tf.reshape(labels['qc'],    [batch_size])

    # if eval_active:

    # pass input through the same network
    net1 = create_inner_model(images1, scope='MobilenetV1', is_training=training_active)
    net2 = create_inner_model(images2, scope='MobilenetV1', is_training=training_active, reuse=True)
    net3 = create_inner_model(images3, scope='MobilenetV1', is_training=training_active, reuse=True)

    with tf.variable_scope('MobilenetV1addon') as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=training_active):

            # concatenate along feature dimension  - 
            net = tf.concat([net1, net2, net3], -1)
            net = slim.conv2d(net, 512, [1, 1])
            net = slim.conv2d(net, 64, [1, 1])
            net = slim.flatten(net) # 64*7*7=3136 features
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net,0.4)
            net_output = slim.fully_connected(net, num_classes)
            #
            logits = tf.contrib.layers.softmax( net_output )

    predictions = {
        'classes': tf.argmax(input=net_output, axis=1),
        'probabilities': logits
    }

    ############ DEBUG ##########
    summary_writer = tf.contrib.summary.create_file_writer(
        os.path.join(params['model_dir'], 'debug'), name='debug')

    with summary_writer.as_default():
        #qc_pass = tf.greater(labels, 0)
        label_1 = tf.equal(labels, 1)

        tf.summary.image("images1", images1)
        # tf.summary.image("images1_pass", tf.boolean_mask(images1, qc_pass))
        # tf.summary.image("images1_fail", tf.boolean_mask(images1, qc_fail))
        # tf.summary.image("images2_pass", tf.boolean_mask(images2, qc_pass))
        # tf.summary.image("images2_fail", tf.boolean_mask(images2, qc_fail))
        # tf.summary.image("images3_pass", tf.boolean_mask(images3, qc_pass))
        # tf.summary.image("images3_fail", tf.boolean_mask(images3, qc_fail))

        tf.summary.histogram( "label_1", tf.boolean_mask(net_output, label_1))
        #tf.summary.histogram( "spatial_score_fail_1", tf.boolean_mask(net[:,:,:,1], qc_fail))
        
        # tf.summary.image("net1_pass", tf.boolean_mask(
        #     net1[:, :, :, 0:1], qc_pass))
        # tf.summary.image("net1_fail", tf.boolean_mask(
        #     net1[:, :, :, 0:1], qc_fail))
        

    if predict_active:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if eval_active and FLAGS.display_tensors and (not params['use_tpu']):
        with tf.control_dependencies([
            tf.Print(
                predictions['classes'], [predictions['classes']],
                summarize=FLAGS.batch_size,
                message='prediction: ')
        ]):
            labels = tf.Print(
                labels, [labels],
                summarize=FLAGS.batch_size, message='label: ')

    one_hot_labels = tf.one_hot(labels, num_classes, dtype=tf.int32)

    tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=logits,
        weights=1.0,
        label_smoothing=0.0)

    loss = tf.losses.get_total_loss(add_regularization_losses=True)
    initial_learning_rate = FLAGS.learning_rate * FLAGS.batch_size / 256
    final_learning_rate = 0.0001 * initial_learning_rate

    train_op = None
    if training_active:
        batches_per_epoch = FLAGS.n_samples // FLAGS.batch_size
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=FLAGS.learning_rate_decay_epochs * batches_per_epoch,
            decay_rate=FLAGS.learning_rate_decay,
            staircase=True)

        # Set a minimum boundary for the learning rate.
        learning_rate = tf.maximum(
            learning_rate,
            final_learning_rate,
            name='learning_rate')

        if FLAGS.optimizer == 'sgd':
            tf.logging.info('Using SGD optimizer')
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif FLAGS.optimizer == 'momentum':
            tf.logging.info('Using Momentum optimizer')
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif FLAGS.optimizer == 'RMS':
            tf.logging.info('Using RMS optimizer')
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON)
        else:
            tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
        # if FLAGS.moving_average:
        #   ema = tf.train.ExponentialMovingAverage(
        #       decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        #   variables_to_average = (tf.trainable_variables() +
        #                           tf.moving_average_variables())
        #   with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
        #     train_op = ema.apply(variables_to_average)

    eval_metrics = None

    if eval_active:

        def metric_fn(labels, predictions):
            return {
                'accuracy': tf.metrics.accuracy(labels, tf.argmax(input=predictions, axis=1)),
                #'auc': tf.metrics.auc(labels, predictions[:, 1])
            }
        eval_metrics = (metric_fn, [labels, logits])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics)


def main(argv):
    del argv  # Unused

    if FLAGS.use_tpu:
        assert FLAGS.model_dir.startswith("gs://"), ("'model_dir' should be a "
                                                     "GCS bucket path!")
        # Resolve TPU cluster and runconfig for this.
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu)
    else:
        tpu_cluster_resolver = None

    batch_size_per_shard = FLAGS.batch_size // FLAGS.num_cores
    batch_axis = 0

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_summary_steps=FLAGS.save_summary_steps,
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations,
            per_host_input_for_training=True))

    inception_classifier = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        config=run_config,
        params={'model_dir': FLAGS.model_dir},
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        batch_axis=(batch_axis, 0))

    def _train_data(params):  # hack ?
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=['deep_qc_data_shuffled_20190805_train.tfrecord'],
            training=True)
        images, labels = dataset.make_one_shot_iterator().get_next()
        return images, labels

    def _eval_data(params):  # hack ?
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=['deep_qc_data_shuffled_20190805_val.tfrecord'],
            training=False)
        images, labels = dataset.make_one_shot_iterator().get_next()
        return images, labels

    eval_hooks = []  # HACK?
    steps_per_cycle = FLAGS.n_samples//FLAGS.batch_size//FLAGS.eval_per_epoch
    #eval_steps     = 2*FLAGS.batch_size

    #eval_steps = 1 if eval_steps<1 else eval_steps

    #print("Training steps:{} Steps per evaluation:{}".format(training_steps,eval_steps))
    for cycle in range(FLAGS.train_epochs * FLAGS.eval_per_epoch):
        #tf.logging.info('Starting training cycle %d.' % cycle)
        inception_classifier.train(
            input_fn=_train_data,
            steps=steps_per_cycle)

        #tf.logging.info('Starting evaluation cycle %d .' % cycle)
        eval_results = inception_classifier.evaluate(
            input_fn=_eval_data,
            hooks=eval_hooks)
        tf.logging.info('Evaluation results: %s' % eval_results)


if __name__ == '__main__':
    # main()
    tf.app.run()
    # tf.compat.v1.app.run()
