# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
from functools import partial
from keras import backend as K
import numpy as np
from keras import losses
        
def tv_loss(y_true, y_pred):
    dx, dy, dz = gradient(y_pred)
    tv_loss = K.mean(K.abs(K.abs(dx))) + \
              K.mean(K.abs(K.abs(dy))) + \
              K.mean(K.abs(K.abs(dz)))
    return tv_loss

def gradient(x):
    assert K.ndim(x) == 5
    if K.image_data_format() == 'channels_first':
        dx = (x[:, :, :-1, :-1, :-1] - x[:, :, 1:, :-1, :-1])
        dy = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, 1:, :-1])
        dz = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, :-1, 1:])
    else:
        dx = (x[:, :-1, :-1, :-1, :] - x[:, 1:, :-1, :-1, :])
        dy = (x[:, :-1, :-1, :-1, :] - x[:, :-1, 1:, :-1, :])
        dz = (x[:, :-1, :-1, :-1, :] - x[:, :-1, :-1, 1:, :])
    return dx, dy, dz

def cross_entropy(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return -(K.sum(K.log(y_pred[y_true==1]))+\
             K.sum(K.log(1 - y_pred[y_true==0])))

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
