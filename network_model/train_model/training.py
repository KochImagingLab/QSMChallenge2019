# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import math
import numpy as np
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback, LambdaCallback
K.set_image_dim_ordering('th')


# ------------------------------------------------------
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
        print('lr:', K.eval(self.model.optimizer.lr))
        
    def on_train_end(self, logs={}):
        pass

# ------------------------------------------------------
class LearningRatePrinter(Callback):
    def __init__(self, model):
        super(LearningRatePrinter, self).__init__()
        self.model = model
    def on_epoch_begin(self, epoch, logs={}):
        print('lr:', K.eval(self.model.optimizer.lr))

# ------------------------------------------------------
class CustomModelCheckpoint(Callback):
    def __init__(self, model, path):
        super(CustomModelCheckpoint, self).__init__()

        self.path = path
        self.best_loss = np.inf
        self.model_for_saving = model    # We set the model (non multi gpu) under an other name
        self.n_steps        = 0

    def on_epoch_end(self, epoch, logs=None):
        #self.model_for_saving.save_weights('model_weight_%d.h5'%(epoch), overwrite=True)
        loss = logs['val_loss']
        print('val_loss:', logs['val_loss'])
        print('best_loss:', self.best_loss)
        if loss<self.best_loss:
            self.model_for_saving.save_weights(self.path, overwrite=True)
            self.best_loss = loss

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model, model_file, weight_file,initial_learning_rate=0.0001, 
                  learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", 
                  verbosity=1, early_stopping_patience=None):
    callbacks = list()
    callbacks.append(CustomModelCheckpoint(model, weight_file))  # Very important using not parallel model to save the weight
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, 
                                                       initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, 
                                                       epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, 
                                           patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    callbacks.append(LearningRatePrinter(model))
    callbacks.append(LearningRateDecay(model, initial_lrate=initial_learning_rate, decay_rate=0.1, n_batch=200))
    return callbacks

def train_model(save_model, model, parallel_model, model_file, weight_file, 
                training_generator, validation_generator, 
                steps_per_epoch, validation_steps,
                n_workers = 4, max_queue_size=4, use_multiprocessing = True,
                initial_learning_rate=0.001, learning_rate_drop=0.5, 
                learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    if parallel_model is None:
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=n_epochs,
                            max_queue_size=max_queue_size,
                            use_multiprocessing=use_multiprocessing,
                            workers=n_workers,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            callbacks=get_callbacks(save_model,
                                                    model_file,
                                                    weight_file,
                                                    initial_learning_rate=initial_learning_rate,
                                                    learning_rate_drop=learning_rate_drop,
                                                    learning_rate_epochs=learning_rate_epochs,
                                                    learning_rate_patience=learning_rate_patience,
                                                    early_stopping_patience=early_stopping_patience))
    else:
        parallel_model.fit_generator(generator=training_generator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=n_epochs,
                                    max_queue_size=max_queue_size,
                                    use_multiprocessing=use_multiprocessing,
                                    workers=n_workers,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps,
                                    callbacks=get_callbacks(save_model,     # Very important using not parallel model to save the weight
                                                            model_file,
                                                            weight_file,
                                                            initial_learning_rate=initial_learning_rate,
                                                            learning_rate_drop=learning_rate_drop,
                                                            learning_rate_epochs=learning_rate_epochs,
                                                            learning_rate_patience=learning_rate_patience,
                                                            early_stopping_patience=early_stopping_patience))        
