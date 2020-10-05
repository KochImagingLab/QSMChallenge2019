# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
from __future__ import absolute_import
from keras import backend as K
from keras import layers
from keras.engine import InputSpec
from keras.layers import Layer
import tensorflow as tf


class NDIErr(Layer):
    def __init__(self, **kwargs):
        super(NDIErr, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5), InputSpec(ndim=5), InputSpec(ndim=5)]

    def build(self, input_shape):
        super(NDIErr, self).build(input_shape)  # Be sure to call this at the end
        self.built = True

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        m  = inputs[2]
        f1d = tf.math.exp(tf.complex(0.0,f1))
        f2d = tf.math.exp(tf.complex(0.0,f2))
        err = tf.abs(tf.multiply(tf.cast(m, tf.complex64), (f1d-f2d)))
        return err

    def compute_output_shape(self, input_shape):
        return (input_shape[0])
