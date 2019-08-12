from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# command line configuration
from tensorflow.python.platform import flags
from model import create_qc_model
import os

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


class LoadEMAHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
      super(LoadEMAHook, self).__init__()
      self._model_dir = model_dir
  
    def begin(self):
      ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
      variables_to_restore = ema.variables_to_restore()
      self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
          tf.train.latest_checkpoint(self._model_dir), variables_to_restore)
  
    def after_create_session(self, sess, coord):
      tf.logging.info('Reloading EMA...')
      self._load_ema(sess)


def _flags2params(flags):
    # HACK
    params = {
        # 'batch_size':flags.batch_size,  # to be set by estimator
        'model_dir':flags.model_dir,
        'moving_average':flags.moving_average,
        'use_tpu':flags.use_tpu,
        'learning_rate':flags.learning_rate,
        'n_samples':flags.n_samples,
        'learning_rate_decay_epochs':flags.learning_rate_decay_epochs,
        'learning_rate_decay':flags.learning_rate_decay,
        'optimizer':flags.optimizer,
        'multigpu':flags.multigpu,
        'gpu':flags.gpu,
        'flavor':flags.flavor
    }
    # for non TPUEstimator need to provide batch_size
    if flags.multigpu or flags.gpu:
        params['batch_size']=flags.batch_size
    
    return params
    

    
def model_fn(features, labels, mode, params):
    num_classes = 2

    training_active = (mode == tf.estimator.ModeKeys.TRAIN)
    eval_active = (mode == tf.estimator.ModeKeys.EVAL)
    predict_active = (mode == tf.estimator.ModeKeys.PREDICT)
    labels = labels['qc']

    net_output, logits, class_out = create_qc_model(features, 
        training_active=training_active,flavor=params.get('flavor','r50'),num_classes=num_classes)

    predictions = {
        'classes': class_out,
        'probabilities': logits
    }

    if predict_active:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    one_hot_labels = tf.one_hot(labels, num_classes, dtype=tf.int32)

    tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=logits,
        weights=1.0,
        label_smoothing=0.0)

    loss = tf.losses.get_total_loss(add_regularization_losses=True)
    initial_learning_rate = params["learning_rate"] * params["batch_size"] / 256
    final_learning_rate = 1e-4 * initial_learning_rate

    train_op = None
    if training_active:
        batches_per_epoch = params["n_samples"] // params["batch_size"]
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=params["learning_rate_decay_epochs"] * batches_per_epoch,
            decay_rate=params["learning_rate_decay"],
            staircase=True)

        # Set a minimum boundary for the learning rate.
        learning_rate = tf.maximum(
            learning_rate,
            final_learning_rate,
            name='learning_rate')

        if params["optimizer"] == 'sgd':
            tf.logging.info('Using SGD optimizer')
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif params["optimizer"] == 'momentum':
            tf.logging.info('Using Momentum optimizer')
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif params["optimizer"] == 'RMS':
            tf.logging.info('Using RMS optimizer')
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON)
        elif params["optimizer"] == 'ADAM': # Doesn't seem to work :(
            tf.logging.info('Using ADAM optimizer')
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        else:
            tf.logging.fatal('Unknown optimizer:', params["optimizer"])

        if params["use_tpu"]:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss)
            if not params["use_tpu"]:
                # TODO: clip gradients
                gradients_norm = tf.linalg.global_norm(gradients,"gradients_norm")
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        if params["moving_average"]:
            ema = tf.train.ExponentialMovingAverage(
                decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
                
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
                 train_op = ema.apply(variables_to_average)

    eval_metrics = None

    if eval_active:
        def metric_fn_ev(_labels, _predictions, _logits):
            return {
                'accuracy': tf.metrics.accuracy(_labels, _predictions),
                'precision': tf.metrics.precision(_labels, _predictions),
                'recall': tf.metrics.recall(_labels, _predictions ),
            }
        eval_metrics = (metric_fn_ev, [labels, class_out, logits])
    else: # do the same
        def metric_fn_tr(_labels, _predictions):
            return {
                'accuracy': tf.metrics.accuracy(_labels, _predictions),
            }
        eval_metrics = (metric_fn_tr, [labels, class_out])

    ############ DEBUG ##########
    #print_op = tf.print(ids)
    if not params["use_tpu"]:
        summary_writer = tf.contrib.summary.create_file_writer(
            os.path.join(params['model_dir'], 'debug' ), name='debug')

        with summary_writer.as_default():
            if training_active:
                #tf.summary.histogram("gradients", gradients)
                with tf.control_dependencies([gradients_norm, labels]): # print_ops
                    tf.summary.scalar("gradient norm", gradients_norm)
    
    if training_active and not params["multigpu"] and not params["gpu"]:
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=None) # eval_metrics
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(labels, tf.argmax(input=net_output, axis=1)),
                'precision': tf.metrics.precision(labels, tf.argmax(input=net_output, axis=1)),
                'recall': tf.metrics.recall(labels, tf.argmax(input=net_output, axis=1)),
                'auc': tf.metrics.auc(labels, logits[:, 1])
            })



def create_AQC_estimator(flags, tpu_cluster_resolver=None,warm_start_from=None):
    session_config = tf.ConfigProto()                                               
    optimizer_options = session_config.graph_options.optimizer_options

    if flags.xla: 
        optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    steps_per_cycle = flags.n_samples*flags.mult//flags.batch_size//flags.eval_per_epoch
    
    if flags.multigpu: # train on multiple GPUs
        _strategy = tf.distribute.MirroredStrategy()
        session_config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = flags.log_device_placement)
        if flags.xla:
            session_config.graph_options.optimizer_options.global_jit_level=tf.OptimizerOptions.ON_1

        run_config = tf.estimator.RunConfig(
            save_checkpoints_secs=flags.save_checkpoints_secs,
            save_summary_steps=flags.save_summary_steps,
            train_distribute=_strategy,
            session_config=session_config
            )
        if flags.testing: # disable saving checkpoints
            run_config.save_checkpoints_steps=None
            run_config.save_checkpoints_secs=None

        aqc_classifier = tf.estimator.Estimator(
            model_fn = model_fn,
            config = run_config,
            model_dir = flags.model_dir,
            params = _flags2params(flags),
            warm_start_from=warm_start_from)
    elif flags.gpu: # train on single GPU using Estimator
        session_config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = flags.log_device_placement)
        if flags.xla:
            session_config.graph_options.optimizer_options.global_jit_level=tf.OptimizerOptions.ON_1

        run_config = tf.estimator.RunConfig(
            save_checkpoints_secs=flags.save_checkpoints_secs,
            save_summary_steps=flags.save_summary_steps,
            session_config=session_config
            )

        if flags.testing: # disable saving checkpoints
            run_config.save_checkpoints_steps=None
            run_config.save_checkpoints_secs=None
        
        aqc_classifier = tf.estimator.Estimator(
            model_fn = model_fn,
            config = run_config,
            model_dir = flags.model_dir,
            params = _flags2params(flags),
            warm_start_from=warm_start_from)

    else: # use TPU estimator
        session_config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = flags.log_device_placement)
            
        if flags.xla:
            session_config.graph_options.optimizer_options.global_jit_level=tf.OptimizerOptions.ON_1

        run_config = tf.estimator.tpu.RunConfig(
            cluster = tpu_cluster_resolver,
            model_dir = flags.model_dir,
            save_checkpoints_secs = flags.save_checkpoints_secs,
            save_summary_steps = flags.save_summary_steps,
            session_config = session_config,
            tpu_config = tf.estimator.tpu.TPUConfig(
                iterations_per_loop = steps_per_cycle,
                per_host_input_for_training = True),
        )
        
        if flags.testing: # disable saving checkpoints
            run_config.save_checkpoints_steps=None
            run_config.save_checkpoints_secs=None

        aqc_classifier = tf.estimator.tpu.TPUEstimator(
            model_fn = model_fn,
            use_tpu = flags.use_tpu,
            config = run_config,
            params = _flags2params(flags),
            eval_on_tpu = False,
            train_batch_size = flags.batch_size,
            eval_batch_size = flags.eval_batch_size,
            warm_start_from = warm_start_from
        ) # batch_axis=(batch_axis, 0)    
    
    return aqc_classifier
