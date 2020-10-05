# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import warnings
import h5py
import keras.backend as K
from keras.models import load_model, model_from_json
from keras import optimizers
from keras.utils import multi_gpu_model

def build_model(model, 
                multi_gpus_training = False, 
                num_gpus = 2,
                optimizer = optimizers.RMSprop(lr=0.0001), 
                loss = 'binary_crossentropy', 
                loss_weight = [1],
                metrics = ['accuracy']
              ):

    if multi_gpus_training and num_gpus<2:
        print("Minimum two GPUs if using multiple GPUs training")
        raise ValueError        

    #-------------------------------
    # Compile model
    #-------------------------------
    if not multi_gpus_training:
        model.compile(loss=loss,
                      loss_weights=loss_weight,
                      optimizer=optimizer,
                      metrics = metrics
                      )
        return model, None

    # Build multi-GPUs model
    else:
        parallel_model = multi_gpu_model(model, gpus=num_gpus)
        parallel_model.compile(loss=loss,
                               loss_weights=loss_weight,
                               optimizer=optimizer,
                               metrics = metrics
                               )
        return model, parallel_model


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

    
def load_model_and_weight(model_file, weight_file):
    try:
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        
        # load weights into new model
        model.load_weights(weight_file, by_name=True)
        print("Loaded model from disk")    
        
        return model
    except ValueError as error:
        raise error             
