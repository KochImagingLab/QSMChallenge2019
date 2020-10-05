# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import os,math
import numpy as np
import nibabel as nib
import keras.backend as K
from keras.callbacks import Callback

def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)

    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)

def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_data(), img.affine

def save_model(model, model_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

def load_model_json(model_file):
    print("Loading pre-trained model")
    try:
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model
    except ValueError as error:
        raise error

def save_weight(model, weight_file):
    model.save_weights(weight_file)

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

def tv_loss(y_true, y_pred):
    dx, dy, dz = gradient(y_pred)
    tv_loss = K.mean(K.abs(dx)) + \
              K.mean(K.abs(dy)) + \
              K.mean(K.abs(dz))
    return tv_loss

class LearningRateDecay(Callback):
    def __init__(self, model, initial_lrate=0.001, decay_rate=0.1, n_batch=100):
        super(LearningRateDecay, self).__init__()
        self.n_steps        = 0
        self.initial_lrate  = initial_lrate
        self.model          = model
        self.decay_rate     = np.float32(decay_rate)
        self.n_batch        = n_batch

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.n_steps += 1
        if self.n_steps%self.n_batch == 0:
            x = self.n_steps//self.n_batch
            lrate = self.initial_lrate * math.exp(-self.decay_rate*x)
            K.set_value(self.model.optimizer.lr, lrate)
            print('lr:', K.eval(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs={}):
        #print('lr:', K.eval(self.model.optimizer.lr))
        pass

    def on_train_end(self, logs={}):
        pass
