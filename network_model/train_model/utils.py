# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import pickle
import os, glob
from random import shuffle
import nibabel as nib
import numpy as np
import h5py

def saveH5(dataIn, fileName):
    if type(dataIn) is dict:
        # write to h5file
        f = h5py.File(fileName, "w")
        for key in dataIn.keys():
            if dataIn[key] is not None:
                dset = f.create_dataset(key, data=dataIn[key]) 
        f.close()
        return 0
    else:
        return -1

def readH5(fileName):
    f = h5py.File(fileName, "r")
    dataOut = {}
    
    keys = [key for key in f.keys()]
    for key in keys:  
        dataOut[key] = f[key][:]
    f.close()    
    return dataOut
    
def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
        
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)  
    
def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_fdata()

# ====================================================================
def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def fetch_data_files(datasets_path, modalities):
    training_data_files = list()

    for subject_dir in glob.glob(os.path.join(datasets_path, "*")):
        subject_files = list()
        files_existing = True

        for modality in modalities:
            if os.path.exists(os.path.join(subject_dir, modality)):
                subject_files.append(os.path.join(subject_dir, modality))
            else:
                files_existing = False

        if files_existing:
            training_data_files.append(tuple(subject_files))
    return training_data_files

def get_validation_split(data_files, training_file, validation_file, data_split=0.8, overwrite=True):
    def split_list(input_list, split=0.8, shuffle_list=True):
        if shuffle_list:
            shuffle(input_list)
        n_training = int(len(input_list) * split)
        training = input_list[:n_training]
        testing = input_list[n_training:]
        return training, testing
    
    if overwrite or not os.path.exists(training_file) or not os.path.exists(validation_file):
        print("Creating validation split...")
        nb_samples = len(data_files)
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split,shuffle_list=True)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

