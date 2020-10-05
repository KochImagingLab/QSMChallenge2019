import numpy as np
import os, glob, shutil
import nibabel as nib
from datetime import datetime

config = {}
config['dataFolder'] = './train_data_cosmos'
config['saveFolder'] = './train_data_cosmos_patches_128x128x128'
config['patch_size'] = [128, 128, 128]
config['overlap_size'] = [32, 32, 32]

def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)

    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)

def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_data(), img.affine

def main(dataFolder, overlap=32):
    if not os.path.exists(config['saveFolder']):
        os.mkdir(config['saveFolder'])

    for root, dirs, files in os.walk(dataFolder):
        for subdir in dirs[:]:
            try:
                iFreq, affine = readNifti(os.path.join(os.path.abspath(root), subdir, "fmap_suscp.nii.gz"))
                suscp, _ = readNifti(os.path.join(os.path.abspath(root), subdir, "suscp.nii.gz"))
                mask = (iFreq!=0)
            except:
                continue             

            # get the bounding box of brain region and then crop it
            d1 = np.max(np.max(mask, axis=1), axis=1)
            d1first = np.nonzero(d1)[0][0]
            d1last = np.nonzero(d1)[0][-1]
            
            d2 = np.max(np.max(mask, axis=0), axis=1)
            d2first = np.nonzero(d2)[0][0]
            d2last = np.nonzero(d2)[0][-1]
            
            d3 = np.max(np.max(mask, axis=0), axis=0)
            d3first = np.nonzero(d3)[0][0]
            d3last = np.nonzero(d3)[0][-1]
            
            iFreqV = iFreq[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            suscpV = suscp[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            maskV  = mask[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]

            nx, ny, nz = iFreqV.shape
            nnx, nny, nnz = config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]
            if nx<nnx:
                iFreqV = np.pad(iFreqV, (((nnx-nx)//2, (nnx-nx)-(nnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, (((nnx-nx)//2, (nnx-nx)-(nnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                suscpV = np.pad(suscpV, (((nnx-nx)//2, (nnx-nx)-(nnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            if ny<nny:
                iFreqV = np.pad(iFreqV, ((0,0),((nny-ny)//2, (nny-ny)-(nny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, ((0,0),((nny-ny)//2, (nny-ny)-(nny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                suscpV = np.pad(suscpV, ((0,0),((nny-ny)//2, (nny-ny)-(nny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            if nz<nnz:
                iFreqV = np.pad(iFreqV, ((0,0),(0,0),((nnz-nz)//2, (nnz-nz)-(nnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, ((0,0),(0,0),((nnz-nz)//2, (nnz-nz)-(nnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                suscpV = np.pad(suscpV, ((0,0),(0,0),((nnz-nz)//2, (nnz-nz)-(nnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))

            nx, ny, nz = iFreqV.shape
            for ix in range(0, nx-config['patch_size'][0]+1, config['overlap_size'][0]):
                for iy in range(0, ny-config['patch_size'][1]+1, config['overlap_size'][1]):
                    for iz in range(0, nz-config['patch_size'][2]+1, config['overlap_size'][2]):
                        iFreqPatch = iFreqV[ix:ix+config['patch_size'][0], iy:iy+config['patch_size'][1], iz:iz+config['patch_size'][2]]
                        suscpPatch = suscpV[ix:ix+config['patch_size'][0], iy:iy+config['patch_size'][1], iz:iz+config['patch_size'][2]]
                        maskPatch  = maskV[ix:ix+config['patch_size'][0], iy:iy+config['patch_size'][1], iz:iz+config['patch_size'][2]]
                    
                        savePath = os.path.join(os.path.abspath(config['saveFolder']),("%s"%(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f'))))
                    
                        if not os.path.exists(savePath):
                            os.mkdir(savePath)
                        else:
                            shutil.rmtree(savePath)
                            os.mkdir(savePath)

                        affMatrix = np.zeros((4,4))
                        affMatrix[0,0] = affine[0,0]
                        affMatrix[1,1] = affine[1,1]
                        affMatrix[2,2] = affine[2,2]
            
                        try:
                            saveNifti(iFreqPatch, os.path.join(savePath, 'fMap.nii.gz'), affMatrix)
                            saveNifti(suscpPatch, os.path.join(savePath, 'suscp.nii.gz'), affMatrix)
                            saveNifti(maskPatch*1,  os.path.join(savePath, 'Mask.nii.gz'), affMatrix)
                        except:
                            continue


if __name__ == "__main__":
    main(config['dataFolder'])
