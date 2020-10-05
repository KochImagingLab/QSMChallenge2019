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
from time import gmtime, strftime
from datetime import datetime
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label, regionprops
from smv import SMV
import random
from scipy.stats import norm

# 04/02 Add light change
# 04/12 Add linear contrast, log contrast, gamma contrast, sigmoid adjust
# change contrast alpha = (np.random.randn(1)[0]*0.1 + 1) -> 0.2


config = {}

config['srcFolder'] = './QSM_2016'
config['saveFolder'] = './train_data_cosmos'
config['voxel_size'] = [1.0, 1.0, 1.0]

config['count'] = 500

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

def Ellipsoid(ellipsoid, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (ellipsoid.shape[0]):
        A   = ellipsoid[k,0]
        asq = ellipsoid[k,1]**2
        bsq = ellipsoid[k,2]**2
        csq = ellipsoid[k,3]**2
        x0  = ellipsoid[k,4]
        y0  = ellipsoid[k,5]
        z0  = ellipsoid[k,6]
        phi = ellipsoid[k,7]
        theta = ellipsoid[k,8]
        psi   = ellipsoid[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = (((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq + (coordp[2,:]-z0)**2/csq) <= 1)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
        
    return p

def Cylinder(cylinder, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (cylinder.shape[0]):
        A   = cylinder[k,0]
        asq = cylinder[k,1]**2
        bsq = cylinder[k,2]**2
        zl  = cylinder[k,3]
        x0  = cylinder[k,4]
        y0  = cylinder[k,5]
        z0  = cylinder[k,6]
        phi = cylinder[k,7]
        theta = cylinder[k,8]
        psi   = cylinder[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = ((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq <= 1) & \
            ((coordp[2,:]-z0)>=-zl) & ((coordp[2,:]-z0)<=zl)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
        
    return p

def Sphere(sphere, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (sphere.shape[0]):
        A   = sphere[k,0]
        asq = sphere[k,1]**2
        bsq = sphere[k,1]**2
        csq = sphere[k,1]**2
        x0  = sphere[k,4]
        y0  = sphere[k,5]
        z0  = sphere[k,6]
        phi = sphere[k,7]
        theta = sphere[k,8]
        psi   = sphere[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = (((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq + (coordp[2,:]-z0)**2/csq) <= 1)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
    
    return p

def Cuboid(cuboid, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (cuboid.shape[0]):
        A  = cuboid[k,0]
        xl = cuboid[k,1]
        yl = cuboid[k,2]
        zl  = cuboid[k,3]
        x0  = cuboid[k,4]
        y0  = cuboid[k,5]
        z0  = cuboid[k,6]
        phi = cuboid[k,7]
        theta = cuboid[k,8]
        psi   = cuboid[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = ((coordp[0,:]-x0)>=-xl) & ((coordp[0,:]-x0)<=xl) & \
            ((coordp[1,:]-y0)>=-yl) & ((coordp[1,:]-y0)<=yl) & \
            ((coordp[2,:]-z0)>=-zl) & ((coordp[2,:]-z0)<=zl)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
    
    return p

# Function to distort image
def elastic_transform(image,alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='constant').reshape(shape)

'''
https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/contrast.py
Gamma contrast ``255*((v/255)**gamma)``, 
Sigmoid contrast ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``, https://en.wikipedia.org/wiki/Sigmoid_function
Log contrast ``255*gain*log_2(1+v/255)``
Linear contrast ``127 + alpha*(v-127)``
'''
def linearContrastAdjust(x, alpha):
    return alpha*x

def gammaContrastAdjust(x, alpha, scale=0.3):
    x += scale/2.0
    return scale*((np.abs(x)/scale)**alpha - (1/2.0)**alpha)

def logContrastAdjust(x, alpha, scale=0.1):
    return np.sign(x)*alpha*scale*(np.log2(1+np.abs(x)/scale))

def sigmoidContrastAdjust(x, gain, scale=0.3, cutoff=0.5):
    x += scale/2.0
    cutoff += np.random.normal(0,0.05,1)[0]
    return scale/(1+np.exp(gain*(cutoff-np.abs(x)/scale))) - scale/(1+np.exp(gain*(cutoff-0.5)))


def main():
    f = open('data.log', 'w')

    if not os.path.exists(config['saveFolder']):
        os.mkdir(config['saveFolder'])

    for count in range(config['count']):
        print('count=%d' % (count))

        try:
            img, affine = readNifti(os.path.join(config['srcFolder'], 'Chi_COSMOS.nii.gz'))
            mask, _ = readNifti(os.path.join(config['srcFolder'], 'Mask.nii.gz'))
            img *= mask
        except:
            print("error reading data", file=f)
            f.flush()
            continue

        # Do elastic transform
        im_merge = elastic_transform(img, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
        
        label_img = label((im_merge!=0), background=0, neighbors=4) 
        propsa = regionprops(label_img) 
        lvs = np.sort(np.bincount(label_img[label_img!=0]))   # count the regions' voxels
        ind = np.argsort(np.bincount(label_img[label_img!=0]))
        mask = (label_img==ind[-1])    
        mask = SMV(mask*1, mask.shape, config['voxel_size'], radius=2)>0.999
        mask_innerbrain = SMV(mask*1, mask.shape, config['voxel_size'], radius=2)>0.999
        
        im_merge = im_merge*mask
      
         
        # Do contrast adjust
        idx = np.random.randint(6, size=1)[0]
        alpha = np.random.normal(0,0.1,1)[0]+1    
        if idx == 0:
            im_merge = linearContrastAdjust(im_merge, alpha)
        elif idx == 1:
            im_merge = gammaContrastAdjust(im_merge, alpha)
        elif idx == 2:
            im_merge = logContrastAdjust(im_merge, alpha)
        elif idx == 3:
            im_merge = sigmoidContrastAdjust(im_merge, alpha+4)
        else:
            pass  

        # Add geometric shapes
        doNo = np.random.randint(6, size=1)[0] 
        if doNo == 0:    
            index = mask.nonzero()
            n = np.random.randint(len(index[0]), size=4) + 1
            ix, iy, iz = index[0][n], index[1][n], index[2][n]
            nx, ny, nz = im_merge.shape[0], im_merge.shape[1], im_merge.shape[2]   
            xs, ys, zs = config['voxel_size'][0], config['voxel_size'][1], config['voxel_size'][2]
            
            bleeds_syn = np.zeros((nx, ny, nz))
            
            for ii in range(len(n)):
                suscp_value = (np.random.randn(1)[0]*0.1 + 2.0)
                
                shape = np.zeros((1,10))
                shape[0,0] = suscp_value
                shape[0,1] = np.random.randint(5, size=1)+1
                shape[0,2] = np.random.randint(5, size=1)+1
                shape[0,3] = np.random.randint(5, size=1)+1
                shape[0,4] = (ix[ii]-nx/2.0)*xs
                shape[0,5] = (iy[ii]-ny/2.0)*ys
                shape[0,6] = (iz[ii]-nz/2.0)*zs
                shape[0,7] = np.random.randint(360,size=1)[0]/180.0*np.pi
                shape[0,8] = np.random.randint(360,size=1)[0]/180.0*np.pi
                shape[0,9] = np.random.randint(360,size=1)[0]/180.0*np.pi                   
                
                idx = np.random.randint(4, size=1)[0] 
                if idx == 0:
                    x = Ellipsoid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                elif idx == 1:
                    x = Sphere(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                #elif idx == 2:
                #    x = Cuboid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                #else:
                #    x = Cylinder(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                else:
                    x = np.zeros([nx, ny, nz])
   
                bleeds_syn = bleeds_syn + x
            if 0 == np.random.randint(2, size=1)[0]:
                bleeds_syn = bleeds_syn*mask
            else:
                bleeds_syn = bleeds_syn*mask_innerbrain
            im_merge = im_merge + bleeds_syn
        elif doNo==1:
            index = mask_innerbrain.nonzero()
            n = np.random.randint(len(index[0]), size=2) + 1
            ix, iy, iz = index[0][n], index[1][n], index[2][n]
            nx, ny, nz = im_merge.shape[0], im_merge.shape[1], im_merge.shape[2]
            xs, ys, zs = config['voxel_size'][0], config['voxel_size'][1], config['voxel_size'][2]

            bleeds_syn = np.zeros((nx, ny, nz))

            for ii in range(len(n)):
                suscp_value = -(np.random.randn(1)[0]*0.1 + 2.0)

                shape = np.zeros((1,10))
                shape[0,0] = suscp_value
                shape[0,1] = np.random.randint(5, size=1)+1
                shape[0,2] = np.random.randint(5, size=1)+1
                shape[0,3] = np.random.randint(5, size=1)+1
                shape[0,4] = (ix[ii]-nx/2.0)*xs
                shape[0,5] = (iy[ii]-ny/2.0)*ys
                shape[0,6] = (iz[ii]-nz/2.0)*zs
                shape[0,7] = np.random.randint(360,size=1)[0]/180.0*np.pi
                shape[0,8] = np.random.randint(360,size=1)[0]/180.0*np.pi
                shape[0,9] = np.random.randint(360,size=1)[0]/180.0*np.pi

                idx = np.random.randint(4, size=1)[0]
                if idx == 0:
                    x = Ellipsoid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                elif idx == 1:
                    x = Sphere(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                #elif idx == 2:
                #    x = Cuboid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                #else:
                #    x = Cylinder(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                else:
                    x = np.zeros([nx, ny, nz])

                bleeds_syn = bleeds_syn + x
                bleeds_syn = bleeds_syn*mask_innerbrain
        
            bleeds_syn = bleeds_syn*mask
            im_merge = im_merge + bleeds_syn
        else:
            im_merge = im_merge
        

        # save data 
        savePath = os.path.join(os.path.abspath(config['saveFolder']),("%s"%(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3])))
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        else:
            shutil.rmtree(savePath)
            os.mkdir(savePath)
        affMatrix = np.zeros((4,4))
        affMatrix[0,0] = config['voxel_size'][0]
        affMatrix[1,1] = config['voxel_size'][1]
        affMatrix[2,2] = config['voxel_size'][2]

        try:
            saveNifti(im_merge*mask, os.path.join(savePath, 'suscp.nii.gz'), affMatrix)
            saveNifti(mask*1, os.path.join(savePath, 'Mask.nii.gz'), affMatrix)
        except:
            continue
   
if __name__ == "__main__":
    main()


