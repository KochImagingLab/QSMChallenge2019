
# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
from __future__ import print_function
import numpy as np
import os, glob, shutil
import nibabel as nib
import keras.backend as K
from keras.engine import Input, Model

config = {}

config['dataFolder'] = './train_data_cosmos'
config['voxel_size'] = [1, 1, 1]
config['matrix_size'] = [160, 160, 160]

def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
    
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)

def readNifti(fileName):
    if not os.path.exists(fileName):
        return None, None
    img = nib.load(fileName)
    return img.get_data(), img.affine

def main(dataFolder):
    count = 0
    f = open('fm.log', 'w')

    Nx = config['matrix_size'][0]
    Ny = config['matrix_size'][1]
    Nz = config['matrix_size'][2]
    voxel_size = config['voxel_size']
    FOV = [Nx*voxel_size[0], Ny*voxel_size[1], Nz*voxel_size[2]]
    kx_squared = np.fft.ifftshift(np.arange(-Nx/2.0, Nx/2.0)/float(FOV[0]))**2
    ky_squared = np.fft.ifftshift(np.arange(-Ny/2.0, Ny/2.0)/float(FOV[1]))**2
    kz_squared = np.fft.ifftshift(np.arange(-Nz/2.0, Nz/2.0)/float(FOV[2]))**2

    [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
    kernel = 1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D)
    kernel[0,0,0] = 0
    
    # --------------------------------------------------
    for root, dirs, files in os.walk(dataFolder):
        for subdir in dirs[:]:
            srcfolder = os.path.join(root, subdir)   #subdir
            print("data - %s" % (srcfolder), file=f)
            f.flush()
                                    
            if os.path.exists(os.path.join(os.path.abspath(dataFolder), subdir, 'fmap_suscp.nii.gz')):
                print("subdir %s - skip" % (subdir), file=f)
                f.flush()
                continue 
            
            if not os.path.exists(os.path.join(os.path.abspath(dataFolder), subdir, 'suscp.nii.gz')):
                print("subdir %s - suscp.nii.gz not exit skip" % (subdir), file=f)
                f.flush()
                #shutil.rmtree(os.path.join(os.path.abspath(dataFolder), subdir))
                continue

            try:
                # load brain susceptibility map
                suscpData3D, aff = readNifti(os.path.join(os.path.abspath(dataFolder), subdir, 'suscp.nii.gz'))
                maskData = (suscpData3D!=0)
              
                # -------------------------------------------
                with K.tf.Session() as sess:
                    fMap = sess.run(K.tf.real(K.tf.ifft3d(K.tf.fft3d(K.tf.complex(suscpData3D, 0.0))*kernel)))
                
                saveNifti(fMap*maskData, os.path.join(os.path.abspath(dataFolder), subdir, 'fmap_suscp.nii.gz'), aff)

                print("subdir %s - done" % (subdir), file=f)
                f.flush()   
            except:
                continue
            # -------------------------------------------
            count += 1
            f.flush()
            
    f.close()
if __name__ == "__main__":
    main(os.path.abspath(config['dataFolder']))

