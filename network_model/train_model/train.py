# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
from __future__ import print_function

import os, glob, sys
import numpy as np
from unet import unet_model_3d
from utils import pickle_dump, pickle_load, fetch_data_files, get_validation_split
from model_utils import save_model, build_model, load_model_and_weight
from training import train_model
from datagenerator import DataGenerator
from keras import optimizers
from metrics import tv_loss

# -------------------------------------------
# Unet settings
config = dict()
config["pool_size"] = (2,2,2)           # pool size for the max pooling operations
config["n_base_filters"] = 24           # num of base kernels
config["conv_kernel"] = (3,3,3)         # convolutional kernel shape
config["layer_depth"] = 5               # unet depth
config["deconvolution"] = False         # if False, will use upsampling instead of deconvolution
config["batch_normalization"] = True    # Using batch norm
config["activation"] = "linear"         # The last convolution layer activation function. For segmentation, using 'sigmoid'.
                                        # For regression, using 'tanh' (if you normalize output to [-1,1]), 'sigmoid' (if you normalize output to [0,1], not good due to gradient vanishing problem) or 'linear'
                                        # please make sure the label data range is within the range of activation function.

# -------------------------------------------
config["image_shape"] = (128,128,128)   # You can use small patch size for training such as (96, 96, 96), (64, 64, 64)
config["input_shape"] = (1,128,128,128)
config["training_modalities"] =  ["fmap_suscp.nii.gz", "Mask.nii.gz"]     # All input image file names
config["output_modalities"] = ["suscp.nii.gz"]                            # Output file name

# -------------------------------------------
# Training settings
config["batch_size"] = 2
config["validation_batch_size"] = config["batch_size"]
config["n_epochs"] = 6               # cutoff the training after this many epochs

config["validation_split"] = 0.9      # portion of the data that will be used for training
config["multi_GPUs_training"] = True  # True - multi-GPUs training, False - single-GPU training
config["num_GPUs"] = 2

# -------------------------------------------
config["datasets_path"] = "./training_data/train_data_cosmos_patches_128x128x128/"  # please change the training data path
config["targetdata_path"] = "./training_data/QSMChallenge2/"
config["target_modalities"] = ["Frequency.nii.gz", "MaskBrainExtracted.nii.gz", "Magnitude.nii.gz"]

config["model_file"] = os.path.abspath("model.h5")                  # save the model structure and weight seperately
config["model_weight_file"] = os.path.abspath("model_weight.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")       # save the training and validation indexs to make sure retraining the saved model training datasets is the same
config["validation_file"] = os.path.abspath("validation_ids.pkl")

# -------------------------------------------
# Training configurations
config["loss"] = ['mae', tv_loss, 'mse']
config["loss_weight"] = [1, 0.05, 20]
config["metrics"] = []

config["patience"] = 10         # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50       # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced

# -------------------------------------------
# Training data generator configurations
# Please refer https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html or check my datagenerator.py
# So that we don't have data generator bottleneck when training, GPU don't wait for CPU data load.
config["n_workers"] = 4
config["max_queue_size"] = 12
config["use_multiprocessing"] = True

# ======================================================================
def main():
    # -------------------------------
    # Multi-GPUs training checking
    if config["multi_GPUs_training"] and config["num_GPUs"]<2:
        print("Error, at least two GPUs when doing multi-GPUs training")

    if config["multi_GPUs_training"]:
        if config["batch_size"]<config["num_GPUs"]:
            config["batch_size"] = config["num_GPUs"]

        if config["validation_batch_size"]<config["num_GPUs"]:
            config["validation_batch_size"] = config["num_GPUs"]        

        batch_per_gpu = config["batch_size"]//config["num_GPUs"]         # make sure the batch_size is dividable by num_gpus
        config["batch_size"] = config["num_GPUs"]*batch_per_gpu

        batch_per_gpu = config["validation_batch_size"]//config["num_GPUs"]
        config["validation_batch_size"] = config["validation_batch_size"]

        print("Do multi-GPUs training")
        print("batch size: %d" % (config["batch_size"]))
        print("validation batch size: %d" % (config["validation_batch_size"]))
        print("Traing data path: %s" % (config["datasets_path"]))

    # -------------------------------
    # get all datasets file names
    data_files = fetch_data_files(config["datasets_path"], config["training_modalities"] + config["output_modalities"])
    print("num of datasets %d" % (len(data_files)))

    # target domain files
    targetdata_files = fetch_data_files(config["targetdata_path"], config["target_modalities"])

    # -------------------------------
    # create data generator for training and validatation, it can load the data from memory pretty fast using multiple workers and buffers if you need to load your data batch by batch
    training_list, validation_list = get_validation_split(data_files,
                                                          config["training_file"],
                                                          config["validation_file"],
                                                          data_split=config["validation_split"])
    # To make sure the num of training and validation cases is dividable by num_gpus when doing multi-GPUs training
    if config["multi_GPUs_training"]:
        num_training_list = len(training_list)//config["batch_size"] * config["batch_size"]
        num_validation_list = len(validation_list)//config["batch_size"] * config["batch_size"]

        training_list = training_list[0:num_training_list]
        validation_list = validation_list[0:num_validation_list]

    training_generator = DataGenerator(data_files,
                                       training_list,
                                       batch_size=config["batch_size"],
                                       shuffle=True,
                                       input_shape=config["image_shape"],
                                       targetdata_files=targetdata_files)
    validation_generator = DataGenerator(data_files,
                                         validation_list,
                                         batch_size=config["batch_size"],
                                         shuffle=False,
                                         input_shape=config["image_shape"],
                                         targetdata_files=targetdata_files)

    print("num of training cases %d" % (len(training_list)))
    print("num of validation cases %d" % (len(validation_list)))
    
    
    # -------------------------------
    # Build neural network structure
    print("Build model")

    unet, unet_t = unet_model_3d(input_shape=config["input_shape"],
                                 pool_size=config["pool_size"],
                                 deconvolution=config["deconvolution"],
                                 depth=config["layer_depth"] ,
                                 n_base_filters=config["n_base_filters"],
                                 kernel = config["conv_kernel"],
                                 batch_normalization=config["batch_normalization"],
                                 activation_name=config["activation"])

    # unet_t is used to save the weights of the model used for target domain

    # save the model s tructure
    # save_model(unet_t, config["model_file"])

    print("unet summary")
    print(unet.summary())

    # ------------------------------
    # compile the model. Please change your optimizer based on your needs
    print("Compile models")
    #optimizer = optimizers.RMSprop(lr=config["initial_learning_rate"], rho=0.9)
    #optimizer = optimizers.SGD(lr=config["initial_learning_rate"], decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = optimizers.Adagrad(lr=config["initial_learning_rate"], epsilon=None, decay=0.0)
    optimizer = optimizers.Adam(lr=config["initial_learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model, parallel_model = build_model(unet,
                                        multi_gpus_training = config["multi_GPUs_training"],
                                        num_gpus = config["num_GPUs"],
                                        optimizer = optimizer,
                                        loss = config["loss"],
                                        loss_weight = config["loss_weight"],
                                        metrics = config["metrics"]
                                        )
    # -------------------------------
    # run training
    train_model(save_model = unet_t, model = model,
                parallel_model = parallel_model,
                model_file  = config["model_file"],
                weight_file = config["model_weight_file"],
                training_generator   = training_generator,
                validation_generator = validation_generator,
                steps_per_epoch  = len(training_list)//config["batch_size"],
                validation_steps = len(validation_list)//config["validation_batch_size"],
                initial_learning_rate = config["initial_learning_rate"],
                learning_rate_drop = config["learning_rate_drop"],
                learning_rate_patience = config["patience"],
                early_stopping_patience= config["early_stop"],
                n_epochs=config["n_epochs"],
                n_workers = config["n_workers"],
                use_multiprocessing = config["use_multiprocessing"],
                max_queue_size = config["max_queue_size"],)

if __name__ == "__main__":
    main()
