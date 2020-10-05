# QSMInvNetExample
# Juan Liu -- Marquette University and
# Kevin Koch -- Medical College of Wisconsin
# Copyright Medical College of Wisconsin, 2020
# 2020
# Please cite using
# Liu, Juan, and Kevin M. Koch. "Non-locally encoder-decoder convolutional network for whole brain QSM inversion." arXiv preprint arXiv:1904.05493 (2019).
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import numpy as np

def sphere_kernel(matrix_size, voxel_size, radius):  
    '''
    Generate a Spherical kernel with the sum normalized to one
    
    Output  
        y - kernel
        
    Input
        matrix_size - the dimension of the field of view
        voxel_size - the size of the voxel in mm
        radius - the raidus of the sphere in mm    
    '''
    
    X, Y, Z = np.mgrid[-matrix_size[0]/2.0:matrix_size[0]/2.0, \
                       -matrix_size[1]/2.0:matrix_size[1]/2.0, \
                       -matrix_size[2]/2.0:matrix_size[2]/2.0]
    
    X *= voxel_size[0]
    Y *= voxel_size[1]
    Z *= voxel_size[2]
    
    Sphere_out =((np.maximum((abs(X)-0.5*voxel_size[0]), 0))**2 + \
                 (np.maximum((abs(Y)-0.5*voxel_size[1]), 0))**2 + \
                 (np.maximum((abs(Z)-0.5*voxel_size[2]), 0))**2)>radius**2
    
    Sphere_in =((abs(X)+0.5*voxel_size[0])**2 + \
                (abs(Y)+0.5*voxel_size[1])**2 + \
                (abs(Z)+0.5*voxel_size[2])**2)<=radius**2   
    
    Sphere_mid = np.zeros(matrix_size)
    
    split = 10
    X_v, Y_v, Z_v = np.mgrid[-split+0.5:split-0.5+1, \
                             -split+0.5:split-0.5+1, \
                             -split+0.5:split-0.5+1]
    
    X_v /= (2.0*split)
    Y_v /= (2.0*split)
    Z_v /= (2.0*split)
    
    shell = 1-Sphere_in-Sphere_out
    Xt = X[shell==1]
    Yt = Y[shell==1]
    Zt = Z[shell==1]
    shell_val = np.zeros(Xt.size)
    
    for i in range(Xt.size):
        occupied = ((Xt[i]+X_v*voxel_size[0])**2+\
                    (Yt[i]+Y_v*voxel_size[1])**2+\
                    (Zt[i]+Z_v*voxel_size[2])**2)<=radius**2
    
        shell_val[i] = 1.0*occupied.sum()/X_v.size
    
    Sphere_mid[shell==1] = shell_val
    Sphere = Sphere_in+Sphere_mid  
    Sphere /= Sphere.sum()  
    y = np.fft.fftn(np.fft.fftshift(Sphere))
    
    return y


def SMV_kernel(matrix_size, voxel_size,radius):
    '''
    Generate a kernel that performs the removal of the spherical mean value
    
    Output
        y - kernel
    
    Input
        matrix_size - the dimension of the field of view
        voxel_size - the size of the voxel in mm
        radius - the raidus of the sphere in mm
    '''
    y = 1.0-sphere_kernel(matrix_size, voxel_size, radius)
    return y

def SMV(iFreq,matrix_size,voxel_size,radius=None):
    '''
    Spherical Mean Value operator
    
    Output
        y - reulstant image after SMV
    input
        iFreq - input image
        matrix_size - dimension of the field of view
        voxel_size - the size of the voxel
        radius - radius of the sphere in mm
    '''
    if radius is None:
        radius = round(6/max(voxel_size)) * max(voxel_size)
        
    skernel = sphere_kernel(matrix_size, voxel_size, radius)
    
    y = np.fft.ifftn(np.fft.fftn(iFreq)*skernel)

    return y
