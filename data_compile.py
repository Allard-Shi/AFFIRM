#!/usr/bin/env python
# coding: utf-8
"""
compile the training, validation and test set

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020
"""

import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
import scipy

def data_compile(data_path_import,
                 data_path_save,
                 data_name,
                 nb_slice,
                 pic_size,
                 seed=2019,
                 inplane_rs=1,
                 nb_cc=4,
                 pic_size_z=192):
        
    """
    compile the training, validation and test set for further processing
    
    including data augmentation by changing the volume size
     
    """
    # set seed
    np.random.seed(seed)
    print('inplane_rs ratio: ', inplane_rs)
    print('start data compilation ...... ')


    data = np.ndarray((0,pic_size,pic_size,nb_slice)) #
    coefficient = []
    ga = []
  
    names = os.listdir(data_path_import)
    n_subject = len(names)
    print('Number of subjects: ', n_subject)
    d_size = pow(np.linspace(0.885,1.125,nb_cc),1/3)
    print(d_size)
    
    for name in tqdm(names):
        
        # the file is named according to GA, e.g., 30_2_xxxxx
        age = int(name[:2])+int(name[3])/7
        
        data_file_name = data_path_import + name
        img_rawdata = nib.load(data_file_name)
        img_data_baseline = img_rawdata.get_data()
        
        # generate images with slightly different sizes
        for cc in range(nb_cc):
            
            inplane_rs =  d_size[cc]
            img_data_rawdata = scipy.ndimage.interpolation.zoom(img_data_baseline,(inplane_rs,inplane_rs,inplane_rs),order=1)
            num_img_slice = img_data_rawdata.shape[2]

            # Select middle slice
            num_valid_pixel = [len(img_data_rawdata[...,i].nonzero()[0]) for i in range(num_img_slice)]
            middle_slice = np.argmax(num_valid_pixel)   
                
            row, col= img_data_rawdata[...,middle_slice].nonzero()
            row_min, row_max = min(row), max(row)
            col_min, col_max = min(col), max(col)
                    
            new_img = img_data_rawdata[row_min:row_max+1, col_min:col_max+1,...]
       
            new_img = new_img.astype(np.float32)
                                
            (img_len, img_wid, _) = new_img.shape
    
            idx_prune = int(np.floor((pic_size_z-nb_slice)/2))
            idx = int(np.floor(pic_size_z/2)-middle_slice)
            img_padding = np.zeros((pic_size,pic_size,pic_size_z)) 
            
            try:
                img_padding[int(np.floor((pic_size-img_len)/2)):int(np.floor((pic_size-img_len)/2)) + img_len
                            ,int(np.floor((pic_size-img_wid)/2)):int(np.floor((pic_size-img_wid)/2)) + img_wid,
                            idx:idx+num_img_slice] = new_img
            except:
                print(name, 'out of the range of image size ip_rs zoom:',inplane_rs)
                continue
            
            img_padding = img_padding[...,idx_prune:-idx_prune]
            img_padding = img_padding.reshape((1,) + img_padding.shape) 
            
            # normalization for network
            img_padding /=np.max(img_padding)
            
            coefficient += [np.max(img_padding)]
            ga += [np.round(age)]
            data = np.concatenate((data,img_padding),axis=0)
    #        label = np.concatenate((label,[float(name[:loc_age])]),axis=0)
    

    final_data = data.reshape((data.shape)+(1,))
    
    np.save(path_save + data_name + '_data.npy', final_data)
    np.save(path_save + data_name + '_coefficient.npy', np.array(coefficient))
    np.save(path_save + data_name + '_ga.npy', np.array(ga))

    print(data_name + ' data saved!') 
    print('compile '+str(len(names))+' subjects done ... ')

if __name__ == '__main__':
        
    # set path
    path_data = r'./data/'
    path_save = r'./compile/'
          
    # select input size
    pic_size = 192
    nb_slice = 144
    inplane_rs = 1
    
    # Compile the data in .npy  
    data_compile(data_path_import=path_data+'train/',
                 data_path_save=path_save,
                 data_name='recon_train',
                 nb_slice=nb_slice,
                 pic_size=pic_size,
                 nb_cc=5)

    data_compile(data_path_import=path_data+'validation/',
                data_path_save=path_save,
                data_name='recon_validation',
                nb_slice=nb_slice,
                pic_size=pic_size,
                nb_cc=5)   

    data_compile(data_path_import=path_data+'test/',
            data_path_save=path_save,
            data_name='recon_test',
            nb_slice=nb_slice,
            pic_size=pic_size,
            nb_cc=5)


