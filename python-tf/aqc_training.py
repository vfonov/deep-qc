# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals

#import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.compat.v1.enable_eager_execution()

BATCH_SIZE = 32
#steps_per_epoch = 200
total_number_of_epochs = 100
training_frac=90
validation_frac=2
testing_frac=8

filenames = [ 'deep_qc_data_0.tfrecord', 'deep_qc_data_1.tfrecord', 'deep_qc_data_2.tfrecord', 'deep_qc_data_3.tfrecord'  ]
raw_ds = tf.data.TFRecordDataset( filenames )

# hardcoded
n_subj=3331
n_samples=57848

# random permutation
all_subjects=np.random.permutation(n_subj)

train_subjects = tf.convert_to_tensor( all_subjects[0:n_subj*training_frac//100] )
validation_subjects = tf.convert_to_tensor(  all_subjects[n_subj*training_frac//100:n_subj*training_frac//100+n_subj*training_frac//100] )
testing_subjects = tf.convert_to_tensor( all_subjects[n_subj*training_frac//100+n_subj*training_frac//100:-1] )

print(train_subjects)

feature_description = {
    'img1_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'img2_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'img3_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'qc':   tf.io.FixedLenFeature([], tf.int64,  default_value=0 ),
    'subj': tf.io.FixedLenFeature([], tf.int64,  default_value=0 )
    }

def _parse_feature(i):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(i, feature_description)

def _decode_jpeg(a):
    img1 = tf.cast(tf.image.decode_jpeg(a['img1_jpeg'], channels=1),dtype=tf.float32)/127.5-1.0
    img2 = tf.cast(tf.image.decode_jpeg(a['img2_jpeg'], channels=1),dtype=tf.float32)/127.5-1.0
    img3 = tf.cast(tf.image.decode_jpeg(a['img3_jpeg'], channels=1),dtype=tf.float32)/127.5-1.0

    return  {'View1':img1, 'View2':img2, 'View3':img3},{'qc':a['qc'], 'subj':a['subj']}


parsed_ds = raw_ds.map(_parse_feature, num_parallel_calls=AUTOTUNE )

# we want to split the database based on subject id's not sample id, since the same subject can be present multiple times
# with slightly different result
training_ds = parsed_ds.filter(lambda x: tf.reduce_any(tf.math.equal(tf.expand_dims(x['subj'],0),tf.expand_dims(train_subjects,1)) )) # hack
training_ds = training_ds.map(_decode_jpeg, num_parallel_calls=AUTOTUNE )
training_ds = training_ds.shuffle(buffer_size=2000) # TODO: determine optimal buffer size
training_ds = training_ds.repeat()
training_ds = training_ds.batch(BATCH_SIZE)
training_ds = training_ds.prefetch(buffer_size=AUTOTUNE)

testing_ds = parsed_ds.filter(lambda x: tf.reduce_any(tf.math.equal(tf.expand_dims(x['subj'],0),tf.expand_dims(testing_subjects,1)) ))
testing_ds = testing_ds.map(_decode_jpeg,  num_parallel_calls=AUTOTUNE )
testing_ds = testing_ds.batch(BATCH_SIZE)

validation_ds = parsed_ds.filter(lambda x: tf.reduce_any(tf.math.equal(tf.expand_dims(x['subj'],0),tf.expand_dims(validation_subjects,1)) ))
validation_ds = validation_ds.map(_decode_jpeg,  num_parallel_calls=AUTOTUNE )
validation_ds = validation_ds.batch(BATCH_SIZE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

# create Keras model
inner_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 1), include_top=False, weights=None)
#inner_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 1), include_top=False, weights=None)

print(inner_model.summary())

# create registration classification model
im1 = layers.Input(shape=(224, 224, 1),name='View1')
im2 = layers.Input(shape=(224, 224, 1),name='View2')
im3 = layers.Input(shape=(224, 224, 1),name='View3')

# use the same inner model for three images
x1 = inner_model(im1) # each will be  6, 6, 1280 ?
x2 = inner_model(im2)
x3 = inner_model(im3)

# join together
x = layers.Concatenate(axis=-1)([x1,x2,x3])

# learn spatial features of the merged layers
x = layers.Conv2D(128,(1,1),activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(16,(1,1),activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(4,(1,1),activation='relu')(x)
x = layers.BatchNormalization()(x)
# output per region activation
x = layers.Conv2D(1,(1,1),activation='sigmoid')(x)
x = layers.AveragePooling2D(pool_size=(7, 7))(x) # average across all image

# end of spatial preprocessing
x = layers.Flatten(name='qc')(x)
#x = layers.Dense(128,activation='relu')(x)
#x = layers.Dense(1,activation='sigmoid',name='qc')(x)
#x = layers.Dense(2)(x)
#x = layers.Activation('softmax',name='qc')(x)

model = tf.keras.models.Model(inputs=[im1,im2,im3], outputs=x)

# true negative rate metric
model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss='binary_crossentropy', # tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy", tf.keras.metrics.AUC()])

print(model.summary())

#tf.keras.utils.plot_model(model, to_file='whole_model.png')


# setup training process
# start training
from datetime import datetime
checkpoint_path = "training/cp.ckpt"

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 mode='max',
                                                 monitor='val_auc',
                                                 verbose=0)
# logging to tensorboard
logdir="runs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    write_grads=True,
    write_images=True,
    update_freq='batch',
    profile_batch=0,
    batch_size=BATCH_SIZE)

# create scheduler

def step_decay_schedule(initial_lr=1e-4, decay_factor=0.9, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return tf.keras.callbacks.LearningRateScheduler(schedule)


lr_sched = step_decay_schedule(initial_lr=1e-5, decay_factor=0.9, step_size=10)

steps_per_epoch=n_samples//BATCH_SIZE

# load model trained previosly
model.load_weights('qc_mobilenet')
# continue training

train_hist=model.fit(training_ds,
  validation_data=validation_ds,
  epochs=total_number_of_epochs,
  steps_per_epoch=steps_per_epoch,
  callbacks=[tensorboard_callback, cp_callback, lr_sched ]  # lr_sched
  )


# save final weights
model.save_weights('qc_mobilenet_s2')

# save whole model
model.save('qc_mobilenet_s2_model.h5')
