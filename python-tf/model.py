# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from nets.resnet_v2 import resnet_v2_50, resnet_v2_152, resnet_v2_200
from nets.mobilenet_v1 import mobilenet_v1_base

# slim tensorflow library
slim = tf.contrib.slim

def _create_inner_model(images, scope=None, is_training=True, reuse=False,flavor='r50'):
    with tf.variable_scope(scope, 'resnet', [images], reuse=reuse ) as _scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
            
            if flavor=='r50':
                net, _ = resnet_v2_50(images, scope=_scope, is_training=is_training, 
                                          global_pool=False,reuse=reuse)
            elif flavor=='r152':
                net, _ = resnet_v2_152(images, scope=_scope, is_training=is_training, 
                                          global_pool=False,reuse=reuse)
            elif flavor=='r200':
                net, _ = resnet_v2_200(images, scope=_scope, is_training=is_training, 
                                          global_pool=False,reuse=reuse)
            elif flavor=='m':
                net, _ = mobilenet_v1_base(images, scope=_scope)
            else:
                # Unknown model
                tf.logging.info('Requested unknown model, giving resnet_v2_50')
                net, _ = resnet_v2_50(images, scope=_scope, is_training=is_training, 
                                          global_pool=False,reuse=reuse)
    return net



def create_qc_model(features, flavor='r50', scope='auto_qc', training_active=True,num_classes=2):
    """Create autoQC model"""

    images1 = features['View1']
    images2 = features['View2']
    images3 = features['View3']

    with tf.variable_scope(scope, 'auto_qc', 
                        [images1,images2,images3]) as _scope:
        net1 = _create_inner_model(images1, scope='InnerModel', is_training=training_active,flavor=flavor)
        net2 = _create_inner_model(images2, scope='InnerModel', is_training=training_active, reuse=True,flavor=flavor)
        net3 = _create_inner_model(images3, scope='InnerModel', is_training=training_active, reuse=True,flavor=flavor)

        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=training_active):
            # concatenate along feature dimension 
            net = tf.concat( [net1, net2, net3], -1)

            # process all views together
            net = slim.conv2d(net, 2*512, [1, 1])
            net = slim.conv2d(net, 32, [1, 1])
            net = slim.conv2d(net, 32, [7,7], padding='VALID') # 7x7 -> 1x1 
            net = slim.conv2d(net, 32, [1,1])

            # flatten here?
            net = slim.dropout(net, 0.5)
            net = slim.conv2d(net, num_classes, [1,1])

            # output heads
            net_output = slim.flatten(net) # -> N,2
            logits = slim.softmax( net_output )
            class_out = tf.argmax(input=net_output, axis=1),

    return net_output, logits, class_out
