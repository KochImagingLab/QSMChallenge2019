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
from keras import optimizers
from unet import unet_model_3d
from utils import saveNifti, readNifti, save_model, load_model_json, save_model, tv_loss, LearningRateDecay
 
config = dict()
config["pool_size"] = (2,2,2)           # pool size for the max pooling operations
config["n_base_filters"] = 24           # num of base kernels
config["conv_kernel"] = (3,3,3)         # convolutional kernel shape
config["layer_depth"] = 5               # unet depth
config["deconvolution"] = False         # if False, will use upsampling instead of deconvolution
config["batch_normalization"] = True    # Using batch norm
config["activation"] = "linear"
config["input_shape"] = (1, 160, 160, 160)

config["voxel_size"] = [1, 1, 1]

config["initial_learning_rate"] = 0.00001 
config["iterations"] = 10

config["loss"] = ['mse', tv_loss]       # data consistency loss + regularization loss
config["model_file"] = os.path.abspath("model.h5")                  # save the model structure and weight seperately
config["model_weight_file"] = os.path.abspath("model_weight.h5")

config["pretrained_weight"] = '/home/jliu/test/train_model_Sim1Snr1/train_model_mag_11/model_weight.h5'
config["loss_weight"] = [1, 0.002]   # you can adjust the loss weight here, for Sim1Snr1 and Sim2Snr1, I used the loss weight [1, 0.002], for Sim1Snr2 and Sim2Snr2, I used the loss weight [1, 0.001]

config["RDF_file"] = "./QSMChallenge2/Sim1Snr1/Frequency.nii.gz"
config["Mask_file"] = "./QSMChallenge2/Sim1Snr1/MaskBrainExtracted.nii.gz"
config["Magnitude_file"] = "./QSMChallenge2/Sim1Snr1/Magnitude.nii.gz"

def main():
    # read the local tissue field, brain mask and magnitude image
    fmap, aff = readNifti(config["RDF_file"])
    mask, _ = readNifti(config["Mask_file"])
    mag, aff = readNifti(config["Magnitude_file"])
    mag = mag[:,:,:,1]  # you can use the magnitude image from other echos as data weighting matrix
    mag /= mag.max()
    mag *= mask
    nxo, nyo, nzo = fmap.shape

    # get the bounding box of brain region then crop and pad the data to desired matrix size
    d1 = np.max(np.max(mask, axis=1), axis=1)
    d1first = np.nonzero(d1)[0][0]
    d1last = np.nonzero(d1)[0][-1]

    d2 = np.max(np.max(mask, axis=0), axis=1)
    d2first = np.nonzero(d2)[0][0]
    d2last = np.nonzero(d2)[0][-1]

    d3 = np.max(np.max(mask, axis=0), axis=0)
    d3first = np.nonzero(d3)[0][0]
    d3last = np.nonzero(d3)[0][-1]

    iFreqV = fmap[d1first:d1last+1,d2first:d2last+1,d3first:d3last+1]
    maskV = mask[d1first:d1last+1,d2first:d2last+1,d3first:d3last+1]
    magV = mag[d1first:d1last+1,d2first:d2last+1,d3first:d3last+1]

    print('%d %d %d' % (iFreqV.shape[0], iFreqV.shape[1], iFreqV.shape[2]))

    nx, ny, nz = iFreqV.shape
    cnnx, cnny, cnnz = config["input_shape"][1], config["input_shape"][2], config["input_shape"][3]

    if nx>cnnx:
        iFreqV = iFreqV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
        maskV  = maskV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
        magV = magV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
    elif nx<cnnx:
        iFreqV = np.pad(iFreqV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        maskV  = np.pad(maskV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        magV = np.pad(magV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
    if ny>cnny:
        iFreqV = iFreqV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
        maskV  = maskV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
        magV = magV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
    elif ny<cnny:
        iFreqV = np.pad(iFreqV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        maskV  = np.pad(maskV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        magV = np.pad(magV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
    if nz>cnnz:
        iFreqV = iFreqV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
        maskV  = maskV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
        magV = magV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
    elif nz<cnnz:
        iFreqV = np.pad(iFreqV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        maskV  = np.pad(maskV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        magV = np.pad(magV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))

    iFreqV = iFreqV[np.newaxis, np.newaxis, :, :, :]
    maskV = maskV[np.newaxis, np.newaxis, :, :, :]
    magV = magV[np.newaxis, np.newaxis, :, :, :]

    # get the dipole kernel in k-space
    Nx, Ny, Nz = cnnx, cnny, cnnz
    FOV = [Nx*config["voxel_size"][0], Ny*config["voxel_size"][1], Nz*config["voxel_size"][2]]
    kx_squared = np.fft.ifftshift(np.arange(-Nx/2.0, Nx/2.0)/float(FOV[0]))**2
    ky_squared = np.fft.ifftshift(np.arange(-Ny/2.0, Ny/2.0)/float(FOV[1]))**2
    kz_squared = np.fft.ifftshift(np.arange(-Nz/2.0, Nz/2.0)/float(FOV[2]))**2

    [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
    kernel = 3*(1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D))
    kernel[0,0,0] = 0
    kernel = kernel[np.newaxis,np.newaxis,:]

    # create the network
    model = unet_model_3d(input_shape=config["input_shape"],
                         pool_size=config["pool_size"],
                         deconvolution=config["deconvolution"],
                         depth=config["layer_depth"] ,
                         n_outputs=1,
                         n_base_filters=config["n_base_filters"],
                         kernel = config["conv_kernel"],
                         batch_normalization=config["batch_normalization"],
                         activation_name=config["activation"])
    # load the model weight
    model.load_weights(config["pretrained_weight"])

    print("model summary")
    print(model.summary())
    save_model(model, config["model_file"])

    # set model optimizer, compile the model, callbacks
    optimizer = optimizers.Adam(lr=config["initial_learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=config["loss"],
                  loss_weights=config["loss_weight"],
                  optimizer=optimizer)    

    callbacks = list()
    callbacks.append(LearningRateDecay(model, initial_lrate=config["initial_learning_rate"], decay_rate=0.1, n_batch=200)) 

    gyro = 42.7747892
    B0 = 7
    iFreqV /= (gyro*B0)

    for i in range(config["iterations"]):
        # ---------------------------------------------------------------------
        # do model prediction and get the QSM
        _, Pred = model.predict([100*iFreqV, maskV, kernel, magV])
        Pred = Pred[0,0]/100.*3

        if nx>cnnx:
            Pred = np.pad(Pred, (((nx-cnnx)//2, (nx-cnnx)-(nx-cnnx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        elif nx<cnnx:
            Pred = Pred[(cnnx-nx)//2:(cnnx-nx)//2+nx,:,:]

        if ny>cnny:
            Pred = np.pad(Pred, ((0,0),((ny-cnny)//2, (ny-cnny)-(ny-cnny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        elif ny<cnny:
            Pred = Pred[:,(cnny-ny)//2:(cnny-ny)//2+ny,:]

        if nz>cnnz:
            Pred = np.pad(Pred, ((0,0),(0,0),((nz-cnnz)//2, (nz-cnnz)-(nz-cnnz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        elif nz<cnnz:
            Pred = Pred[:,:,(cnnz-nz)//2:(cnnz-nz)//2+nz]

        chi_pred = np.zeros((nxo, nyo, nzo))
        chi_pred[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1] = Pred

        saveNifti(chi_pred, 'chi_iter_%d.nii.gz'%(i*100))

        # ---------------------------------------------------------------------
        # model fine-tuning, run 100 iterations every time
        model.fit([100*iFreqV, maskV, kernel, magV], [np.zeros(iFreqV.shape), np.zeros(iFreqV.shape)], epochs=100, batch_size=1, callbacks=callbacks, verbose=1)
        #model.save_weights('model_weight_%d.h5'%(i), overwrite=True)


 
if __name__ == "__main__":
    main()
