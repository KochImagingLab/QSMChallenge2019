# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, \
                         LeakyReLU, Deconvolution3D, Multiply, Lambda, Add, Subtract, Dense, Reshape
from keras.optimizers import Adam
from keras.models import Sequential
from fmLayer import CalFMLayer
from NDI import NDIErr

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate
      
# 3D UNET, please refer http://deeplearning.net/tutorial/unet.html 
# We keep the image size of input/output

def unet_model_3d(input_shape, 
                  pool_size=(2, 2, 2), 
                  n_outputs=1, 
                  deconvolution=True,
                  kernel=(3,3,3),
                  depth=5, 
                  n_base_filters=32, 
                  batch_normalization=True, 
                  activation_name="linear"):

    # Input - field Map
    fm_in1 = Input((1, input_shape[1], input_shape[2], input_shape[3]))
    mask1  = Input((1, input_shape[1], input_shape[2], input_shape[3]))

    fm_in2 = Input((1, input_shape[1], input_shape[2], input_shape[3]))
    mask2  = Input((1, input_shape[1], input_shape[2], input_shape[3]))
    qsm_kernel = Input((1, input_shape[1], input_shape[2], input_shape[3]))
    ww = Input((1, input_shape[1], input_shape[2], input_shape[3]))

    current_layer1 = concatenate([fm_in1, mask1], axis=1)
    current_layer2 = concatenate([fm_in2, mask2], axis=1)
    levels1 = list()
    levels2 = list()

    for layer_depth in range(depth):
        layer1, layer2 = create_convolution_block(current_layer1, current_layer2,
                                                  kernel = kernel,
                                                  n_filters=n_base_filters*(2**layer_depth),
                                                  batch_normalization=batch_normalization)

        if layer_depth < depth - 1:
            current_layer1 = MaxPooling3D(pool_size=pool_size)(layer1)
            levels1.append([layer1, current_layer1])
            current_layer2 = MaxPooling3D(pool_size=pool_size)(layer2)
            levels2.append([layer2, current_layer2])
        else:
            current_layer1 = layer1
            levels1.append([layer1])
            current_layer2 = layer2
            levels2.append([layer2])
   

    for layer_depth in range(depth-2, -1, -1):
        up_conv = get_up_convolution(pool_size=pool_size,
                                                kernel_size=pool_size,
                                                deconvolution=deconvolution,
                                                n_filters=current_layer1._keras_shape[1]/2)
        up_convolution1 = up_conv(current_layer1)
        up_convolution2 = up_conv(current_layer2)
        concat1 = concatenate([up_convolution1, levels1[layer_depth][0]], axis=1)
        concat2 = concatenate([up_convolution2, levels2[layer_depth][0]], axis=1)
        current_layer1, current_layer2 = create_convolution_block(concat1, concat2, 
                                                 n_filters=levels1[layer_depth][0]._keras_shape[1],
                                                 kernel=kernel,
                                                 batch_normalization=batch_normalization,
                                                 dilation_rate=(1,1,1))

    conv = Conv3D(n_outputs, kernel, padding='same')
    out1= conv(current_layer1)
    out2= conv(current_layer2)
    out1 = Activation(activation_name)(out1)
    out1 = Multiply()([out1, mask1])
    out2 = Activation(activation_name)(out2)
    out2 = Multiply()([out2, mask2])

    fm2 = CalFMLayer()([out2, qsm_kernel])
    err_fm = NDIErr()([fm_in2, fm2, ww])
    err_fm = Multiply()([err_fm, mask2])

    #model = Model(inputs=[fm_in1, mask1, fm_in2, mask2, qsm_kernel],
    #              outputs=[out1, out2, err_fm])

    model_t = Model(inputs=[fm_in2, mask2, qsm_kernel, ww],
                  outputs=[err_fm, out2])
    return model_t 

def create_convolution_block(input_layer1,input_layer2,n_filters, batch_normalization=True, 
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), dilation_rate=(1,1,1)):
    op = Conv3D(n_filters, kernel, padding=padding, strides=strides, dilation_rate=dilation_rate)
    layer1 = op(input_layer1)
    layer2 = op(input_layer2)

    if batch_normalization:
        layer1 = BatchNormalization(axis=1, momentum=0.8)(layer1)
        layer2 = BatchNormalization(axis=1, momentum=0.8)(layer2)
    
    if activation is None:
        return Activation('relu')(layer1), Activation('relu')(layer2)
    else:
        return Activation(activation)(layer1), Activation(activation)(layer2)

def get_up_convolution(n_filters, pool_size, kernel_size=(2,2,2), strides=(2, 2, 2), deconvolution=True):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, 
                               padding = 'same',
                               kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
