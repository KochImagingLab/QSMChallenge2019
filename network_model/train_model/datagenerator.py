# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
import numpy as np
import os, keras
from random import shuffle
import nibabel as nib
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_files, training_list, targetdata_files, batch_size=32, shuffle=True, input_shape=[128,128,128]):
        'Initialization'
        self.data_files = data_files
        self.indexes    = training_list
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.targetdata_files = targetdata_files
        
        if np.remainder(len(self.indexes), batch_size) > 0:
            num_batches = len(self.indexes)//batch_size
            self.indexes = self.indexes[0:num_batches*batch_size]
            
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        # X - local field of synthetic training data
        # m - brain mask of synthetic training data
        # y - susceptibility label of synthetic training data
        # X1 - local field of target domain data
        # m1 - brain mask of target domain data
        # d - dipole kernel
        # w - data weighting matrix of target domain data
        X,m,y,X1,m1,d,w = self.__data_generation(indexes)

        return [X,m,X1,m1,d,w], [y, np.zeros(m.shape), np.zeros(m.shape)]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        y_list = list()
        m_list = list()
        d_list = list()
        xx_list = list()
        mm_list = list()
        mag_list = list()

        # Generate data
        for i, index in enumerate(indexes):
            image_list = list()
            for k, image_file in enumerate(self.data_files[index]):
                image = nib.load(os.path.abspath(image_file))
                image_list.append(image)
    
            subject_data = [image.get_data() for image in image_list]
            
            # x - local tissue field, m - brain mask, y - susceptibility label from synthetic data
            x = np.asarray(subject_data[0])[np.newaxis,:]
            m = np.asarray(subject_data[1])[np.newaxis,:]
            y = np.asarray(subject_data[2])[np.newaxis,:]                        

            #x = x[:, 16:16+128, 16:16+128, 16:16+128]
            #m = m[:, 16:16+128, 16:16+128, 16:16+128]
            #y = y[:, 16:16+128, 16:16+128, 16:16+128]
 
            # x1 - local tissue field, m1 - brain mask, mag1 - magnitude image. I used the magnitude image of second echo as data weighting matrix.
            # For QSMChallenge2, each datasets have different SNR and contrast, I trained individual network.
            # Here the code to train only one network to save the time. Each batch, randomly read one of datasets.
            # Strictly, it is not a good way since it mixed the training and validation data.
            
            #x1 = nib.load("/home/jliu/test/QSMChallenge2/Sim1Snr1/Frequency.nii.gz").get_data()[np.newaxis,:]
            #m1 = nib.load("/home/jliu/test/QSMChallenge2/Sim1Snr1/MaskBrainExtracted.nii.gz").get_data()[np.newaxis,:]
            #mag1 = nib.load("/home/jliu/test/QSMChallenge2/Sim1Snr1/Magnitude.nii.gz").get_data()[:,:,:,1][np.newaxis,:]
            
            i = np.random.randint(len(self.targetdata_files), size=1)[0]
            x1 = nib.load(os.path.join(self.targetdata_files[i][0])).get_data()[np.newaxis,:]
            m1 = nib.load(os.path.join(self.targetdata_files[i][1])).get_data()[np.newaxis,:]
            mag1 = nib.load(os.path.join(self.targetdata_files[i][2])).get_data()[:,:,:,1][np.newaxis,:]
            mag1 /= mag1.max()

            # the belowing code is to crop the x1, m1, mag1 to match the shape of input data for network training
            d1 = np.max(np.max(m1[0,:,:,:], axis=1), axis=1)
            d1first = np.nonzero(d1)[0][0]
            d1last = np.nonzero(d1)[0][-1]

            d2 = np.max(np.max(m1[0,:,:,:], axis=0), axis=1)
            d2first = np.nonzero(d2)[0][0]
            d2last = np.nonzero(d2)[0][-1]

            d3 = np.max(np.max(m1[0,:,:,:], axis=0), axis=0)
            d3first = np.nonzero(d3)[0][0]
            d3last = np.nonzero(d3)[0][-1]

            x1 = x1[:,d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            m1 = m1[:,d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            mag1 = mag1[:,d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]

            _, nxx, nyy, nzz = x1.shape
            ix,iy,iz = 0,0,0
            if nxx>self.input_shape[0]:
                ix = np.random.randint(nxx-self.input_shape[0], size=1)[0]
            if nyy>self.input_shape[1]:
                iy = np.random.randint(nyy-self.input_shape[1], size=1)[0]
            if nzz>self.input_shape[2]:
                iz = np.random.randint(nzz-self.input_shape[2], size=1)[0]
            x1 = x1[:, :, :, iz:iz+self.input_shape[0]]
            m1 = m1[:, :, :, iz:iz+self.input_shape[1]]
            mag1 = mag1[:, :, :, iz:iz+self.input_shape[2]]
 
            _, nx, ny, nz = x1.shape
            cnnx, cnny, cnnz = self.input_shape[0], self.input_shape[1], self.input_shape[2]

            if nx>cnnx:
                x1 = x1[:,(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                m1 = m1[:,(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                mag1 = mag1[:,(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
            elif nx<cnnx:
                x1 = np.pad(x1, ((0,0),((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                m1 = np.pad(m1, ((0,0),((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                mag1 = np.pad(mag1, ((0,0),((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
            if ny>cnny:
                x1 = x1[:,:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                m1 = m1[:,:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                mag1 = mag1[:,:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
            elif ny<cnny:
                x1 = np.pad(x1, ((0,0),(0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                m1 = np.pad(m1, ((0,0),(0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                mag1 = np.pad(mag1, ((0,0),(0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
            if nz>cnnz:
                x1 = x1[:,:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                m1 = m1[:,:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                mag1 = mag1[:,:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
            elif nz<cnnz:
                x1 = np.pad(x1, ((0,0),(0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                m1 = np.pad(m1, ((0,0),(0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
                mag1 = np.pad(mag1, ((0,0),(0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

            # get the dipole kernel
            voxel_size = [1,1,1]
            FOV = [cnnx*voxel_size[0], cnny*voxel_size[1], cnnz*voxel_size[2]]
            kx_squared = np.fft.ifftshift(np.arange(-cnnx/2.0, cnnx/2.0)/float(FOV[0]))**2
            ky_squared = np.fft.ifftshift(np.arange(-cnny/2.0, cnny/2.0)/float(FOV[1]))**2
            kz_squared = np.fft.ifftshift(np.arange(-cnnz/2.0, cnnz/2.0)/float(FOV[2]))**2

            [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
            kernel = 3*(1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D))
            kernel[0,0,0] = 0
            kernel = kernel[np.newaxis,:]

            # the local tissue field is scaled in the real data
            x *= m
            y *= m
            gyro = 42.7747892
            B0 = 7
            x1 /= (gyro*B0)
            x1 *= m1
            mag1 *= m1

            # Store sample
            x_list.append(x)
            m_list.append(m)
            y_list.append(y) 
            xx_list.append(x1)
            mm_list.append(m1)
            d_list.append(kernel)
            mag_list.append(mag1)

        return 100*np.asarray(x_list), np.asarray(m_list), 100/3.0*np.asarray(y_list), 100*np.asarray(xx_list), np.asarray(mm_list), np.asarray(d_list), np.asarray(mag_list)
