from __future__ import absolute_import
from keras import backend as K
from keras import layers
from keras.engine import InputSpec
from keras.layers import Layer
import tensorflow as tf


class CalFMLayer(Layer):
    def __init__(self, **kwargs):
        super(CalFMLayer, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5), InputSpec(ndim=5)]

    def build(self, input_shape):
        super(CalFMLayer, self).build(input_shape)  # Be sure to call this at the end
        self.built = True

    def call(self, inputs):
        suscp = inputs[0]
        kernel = inputs[1]
        ks = K.tf.fft3d(tf.cast(suscp, tf.complex64))
        ks = ks*tf.cast(kernel,tf.complex64)
        fm = K.tf.real(K.tf.ifft3d(ks))     
        return fm

    def compute_output_shape(self, input_shape):
        return (input_shape[0])
