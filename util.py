#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

utils for AFFIRM

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020

"""
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage import transform

def rotate_matrix(theta=[0,0,0], ort='axi'):
    """
    Argument:
        theta: three rotation angle [x,y,z]
        order: the order in idx
        
        
    return 4*4 rotation matrix Rx-Ry-Rz-location
    
    """    
    theta_x = np.deg2rad(theta[0])
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = np.deg2rad(theta[1])
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    theta_z = np.deg2rad(theta[2])
    cz = np.cos(theta_z)
    sz = np.sin(theta_z)
           
    Rx = [[1, 0, 0, 0],
          [0, cx, -sx, 0],
          [0, sx, cx, 0,],
          [0, 0, 0, 1]]

    Ry = [[cy, 0, sy, 0],
          [0, 1, 0, 0],
          [-sy, 0, cy, 0],
          [0, 0, 0, 1]]
        
    
    Rz = [[cz, -sz, 0, 0],
          [sz, cz, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
    
    rot = np.matmul(Rx, Ry)
    rot = np.matmul(rot, Rz)
    
    # define three standard orientations
    if ort == 'axi':
        ort_rot = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        
    elif ort == 'cor':
        ort_rot = [[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]
        
    elif ort == 'sag':
        ort_rot = [[0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 0, 1]]
    
    else:
        raise ValueError('Orientaion notation is invalid! Should be axi or sag or cor. Got {}'.format(ort))
    
    rot = np.matmul(rot,ort_rot)
    
    return rot
  
def matrix2vec(matrix):
    """
    affine matrix (4*4) to affine param vector
    
    """
    vector = np.zeros((1,6))

    rot = matrix[:3,:3]
                       
    angle_y = np.arcsin(rot[0,2])
    angle_x = np.arcsin(-rot[1,2]/np.cos(angle_y))  
    angle_z = np.arcsin(-rot[0,1]/np.cos(angle_y))  
    
    vector[:,3] = np.rad2deg(angle_x) 
    vector[:,4] = np.rad2deg(angle_y)
    vector[:,5] = np.rad2deg(angle_z)
    
    return vector
       
def ancpoint2euler(anchor_point, z_loc):
    """
    
    The function that transforms anchor point representation to angle as well as displacement
    modified from http://github.com/farrell236/SVRnet
    
    Note: the initial anchor point should be set as 
    [ap_1 (-L,-L,z0), ap_2 (0,0,z0), ap_3 (L,-L,z0)]
    
    Hou, Benjamin, et al. "3-D reconstruction in canonical co-ordinate space 
    from arbitrarily oriented 2-D images." 
    IEEE transactions on medical imaging 37.8 (2018): 1737-1750.
         
    """      
    ap_1, ap_2, ap_3 = anchor_point
    
#    print('disp:',disp)
    v1 = ap_3 - ap_1
    v2 = ap_2 - ap_1
    n1 = np.cross(v1, v2)
    n2 = np.cross(n1, v1)

    v1_norm = v1 / np.linalg.norm(v1)  
    n2_norm = n2 / np.linalg.norm(n2)  
    n1_norm = n1 / np.linalg.norm(n1)
    
    # calculate 
    rot = np.array([[v1_norm[0],n2_norm[0],n1_norm[0]],
                    [v1_norm[1],n2_norm[1],n1_norm[1]],
                    [v1_norm[2],n2_norm[2],n1_norm[2]]])
                   
    disp_x, disp_y, disp_z = ap_2[0]-n1_norm[0]*z_loc, ap_2[1]-n1_norm[1]*z_loc, ap_2[2]-n1_norm[2]*z_loc

    angle_y = np.rad2deg(np.arcsin(-rot[0,2]))
    angle_x = np.rad2deg(np.arcsin(rot[1,2]/np.cos(np.deg2rad(angle_y))))
    angle_z = np.rad2deg(np.arcsin(-rot[0,1]/np.cos(np.deg2rad(angle_y))))
       
#    ap_1 = ap_1/np.linalg.norm(ap_1)
#    ap_1_org = ap_1_org/np.linalg.norm(ap_1_org)
#    
#    k = np.cross(ap_1_org,ap_1)   
#    k = k/np.linalg.norm(k)
#    theta = np.arccos(np.dot(ap_1.T,ap_1_org))    
#    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
#    k = k.reshape((3,1))
#    
#    # got rotation matrix
#    rot = np.cos(theta)*np.identity(3) + (1-np.cos(theta))*np.dot(k,k.T) + np.sin(theta)*K

    angle = np.array([angle_x, angle_y, angle_z])    
    disp = np.array([disp_x, disp_y, disp_z])
    rigid_params  = np.concatenate((disp,angle))
    
    return rigid_params 
    
    
def displacement_matrix(disp=[0,0,0], pixrecon=0.8):
    """
    compute displacement matrix (addictive)
    
    """
    
    disp_matrix = np.zeros((4,4))
    disp_matrix[0,3] += disp[0] * pixrecon
    disp_matrix[1,3] += disp[1] * pixrecon
    disp_matrix[2,3] += disp[2] * pixrecon
    
    return disp_matrix

def nii_affine(pixdim, pixrecon, vol_shape, 
               ort='axi', params=[0,0,0,0,0,0]):
    """
    export the Nifty affine information and transformation info.
     
    """       
    # scale matrix
    scale_matrix = [[pixdim[0], 0, 0, 0],
                    [0, pixdim[1], 0, 0],
                    [0, 0, pixdim[2], 0],
                    [0, 0, 0, 1]]
        
    # displacement
    standard_matrix = [[1, 0, 0, -(vol_shape[0]-1)/2],
                       [0, 1, 0, -(vol_shape[1]-1)/2],
                       [0, 0 ,1, -(vol_shape[2]-1)/2],
                       [0, 0, 0, 1]]
       
    st_crd = np.matmul(scale_matrix, standard_matrix)    
    rot = rotate_matrix(params[3:], ort=ort)
    disp = displacement_matrix(params[:3], pixrecon)
     
    affine = disp + np.matmul(rot,st_crd)
            
    return affine

def motion_affine(motion_corrupt=[0,0,0,0,0,0], global_offset=[0,0,0,0,0,0], ort='axi', pixrecon=0.8):
    """
    calculate the motion transformation parameter 
    
    Note: there is a coordinate transform between SimpleITK and the 
          physical coordinate  

    """
    # coordinate transform as sitk follow [z,y,x]
    A = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    
    rot_all = rotate_matrix(motion_corrupt[3:])
    disp_all = displacement_matrix(motion_corrupt[:3], pixrecon)   
    glb_all = disp_all + rot_all
    glb_all = np.matmul(A,glb_all)

    rot_offset = rotate_matrix(global_offset[3:])
    disp_offset = displacement_matrix(global_offset[:3], pixrecon)    
    glb_offset = disp_offset + rot_offset
    glb_offset = np.matmul(A,glb_offset)
   
    rigid_motion = np.matmul(glb_all, np.linalg.inv(glb_offset))  
    motion_rot = rigid_motion[:3,:3]
    motion_disp = rigid_motion[:3,3]

    return motion_rot, motion_disp
    
def save_data(data, save_path, name, 
              params=[0.,0.,0.,0.,0.,0.,], 
              ort='axi', mask=True, 
              pixdim=[0.8,0.8,4.], verbose=False):
    """
    data should be normalized from 0 to 1
    
    Arguments:
        - params [theta_x, theta_y, theta_z, 
                  disp_x,  disp_y,  disp_z]
    
    """
    vol_shape = data.shape
    mask = np.zeros(vol_shape)
    mask[data > 0] = 1
    
    assert len(vol_shape) == 3, 'input should be one 3D volume!'
           
    img_affine = nii_affine(pixdim, pixrecon=0.8, vol_shape=vol_shape, 
                            ort=ort, params=params)
    
    empty_header = nib.Nifti1Header()       
    img = nib.Nifti1Image(data, img_affine, empty_header)    
    
    # set scanner coordinate
    img.header['sform_code'] = 1
    
    # define the standard space as 1*1*1 mm3
    img.header['xyzt_units'] = 10
    
    # set pixel dim
    img.header['pixdim'] = [-1.] + pixdim + [0., 0., 0. , 0.]  
    
    mask = nib.Nifti1Image(mask, img.affine, img.header)  

    if verbose:
        print(img.affine)   
    
    nib.save(img, save_path+name+'.nii.gz')
        
    if mask:
        
        nib.save(mask, save_path+name+'_mask.nii.gz')        
        print('--- ' + name + ' saved!')
        
        return True

    else:
        
        print('save without mask!')
        return True
    
    
def save_mc(path, rigid_transform, global_params, ort, slice_idx, pixdim_recon=0.8):
    """
    # save motion correction parameters (rigid/affine)
    
    a small interface with regards with NiftyMIC reconstruction 
    
    rigid_transform [6,]
    global_params list 6 [dis,dis,dis,rot,rot,rot]
    

    """
    motion_rot, motion_disp = motion_affine(rigid_transform, global_params, 
                                            pixrecon=pixdim_recon, ort='axi')
    
    motion_rot = np.ndarray.flatten(motion_rot)
    rigid_transform_sitk = sitk.AffineTransform(3)
    rigid_transform_sitk.SetMatrix(motion_rot)
    rigid_transform_sitk.SetTranslation(motion_disp)
    path_transform = os.path.join(path,'%s_slice%d.tfm'%(ort,slice_idx))
    sitk.WriteTransform(rigid_transform_sitk, path_transform)
    
    return True
    
def CenterAlignment(img_rawdata, 
                    inplane_shape,
                    throughplane_depth,
                    save_path,
                    save_name,
                    ort='axi',
                    pre_rot=0,
                    nb_slice=192,
                    rotation=0,
                    interleave=2,
                    init_idx=1,
                    pixdim=[0.8]*3,
                    automatic=True,
                    verbose=False):
    """
    
    function that performs center alignment
    
    can be replaced by DL-based centeralignment which is not included in this file
    
    TODO: 
        
    """
    center = int(inplane_shape[0]/2)
    pic_size_z = int(interleave*nb_slice)
    num_img_slice = img_rawdata.shape[2]    
    
    idx_slice = [interleave*i+init_idx for i in range(int(num_img_slice/interleave))]
        
    img_rawdata = img_rawdata[...,idx_slice]
        
    num_valid_pixel = [len(img_rawdata[...,i].nonzero()[0]) for i in range(len(idx_slice))]
    middle_slice = np.argmax(num_valid_pixel)   
    
    if verbose:
        print(num_valid_pixel, middle_slice)
    
    row, col = img_rawdata[...,middle_slice].nonzero()
    row_min, row_max = min(row), max(row)
    col_min, col_max = min(col), max(col)
            
    new_img = img_rawdata[row_min:row_max+1,col_min:col_max+1,:]
   
    new_img = new_img.astype(np.float32)
                        
    (img_len, img_wid, _) = new_img.shape

    idx_prune = int((pic_size_z-nb_slice)/2)
    idx = int((pic_size_z/2-middle_slice))
    img_padding = np.zeros((*inplane_shape,pic_size_z)) 
    
    try:
        img_padding[int(np.floor((inplane_shape[0]-img_len)/2)):int(np.floor((inplane_shape[0]-img_len)/2)) + img_len,
                    int(np.floor((inplane_shape[1]-img_wid)/2)):int(np.floor((inplane_shape[1]-img_wid)/2)) + img_wid,
                    idx:idx+len(idx_slice)] = new_img
    except:
        
        print('the number of slice is not adequate for the given volume!')
        
        
    img_padding = img_padding[...,idx_prune:idx_prune+nb_slice]
    num_valid_pixel = [len(img_padding[...,i].nonzero()[0]) for i in range(nb_slice)]
    middle_slice = np.argmax(num_valid_pixel)   
    
    if verbose:
        print('ckeck middle slice',middle_slice)


    if automatic:
        # optimize the reorientation algorithm
        if ort == 'axi':
            sum_lower = np.sum(num_valid_pixel[:middle_slice])
            sum_upper = np.sum(num_valid_pixel[middle_slice:])
    
#            print('lower-upper:',sum_lower,sum_upper)
            
            if sum_lower > sum_upper:
                img_padding = np.rot90(img_padding,2,axes=[1,2])
                
            sum_front = len(img_padding[:center,:,:].nonzero()[0])
            sum_back = len(img_padding[center:,:,:].nonzero()[0])
            sum_left = len(img_padding[:,:center,:].nonzero()[0])
            sum_right = len(img_padding[:,center:,:].nonzero()[0])  
            rot = np.argmax([sum_left,sum_front,sum_right,sum_back])
            img_padding = np.rot90(img_padding,rot,axes=[0,1])
            
        elif ort == 'cor':
            sum_lower = len(img_padding[...,middle_slice-6:middle_slice].nonzero()[0])
            sum_upper = len(img_padding[...,middle_slice:middle_slice+6].nonzero()[0])
         
            if sum_lower > sum_upper:
                img_padding = np.rot90(img_padding,2,axes=[1,2])
                
            sum_fl = len(img_padding[:center,:center,:].nonzero()[0])
            sum_bl = len(img_padding[center:,:center,:].nonzero()[0])
            sum_fr = len(img_padding[:,:center,:].nonzero()[0])
            sum_br = len(img_padding[:,center:,:].nonzero()[0])    
                   
            sum_f = sum_fl+sum_fr
            sum_b = sum_bl+sum_br
            sum_l = sum_fl+sum_bl
            sum_r = sum_fr+sum_br
            
            rot = np.argmin([sum_l,sum_b,sum_r,sum_f])
            
            img_padding = np.rot90(img_padding,rot,axes=[0,1])
        
        elif ort == 'sag':
            
            sum_f = len(img_padding[:center,...].nonzero()[0])
            sum_b = len(img_padding[center:,...].nonzero()[0])
            sum_l = len(img_padding[:,:center,:].nonzero()[0])
            sum_r = len(img_padding[:,center:,:].nonzero()[0])
            sum_fl = len(img_padding[:center,:center,:].nonzero()[0])
            sum_bl = len(img_padding[center:,:center,:].nonzero()[0])
            sum_fr = len(img_padding[:,:center,:].nonzero()[0])
            sum_br = len(img_padding[:,center:,:].nonzero()[0])    
        
            rot = np.argmax([sum_fr,sum_fl,sum_bl,sum_br])    
            img_padding = np.rot90(img_padding,rot,axes=[0,1])
        
        else:
            raise('ORT is not valid! got {}'.format(ort))
        
    # finetune the orientation
    img_padding = np.rot90(img_padding,pre_rot,axes=[1,2])
    if rotation != 0:
        for i in range(nb_slice):
            img_padding[...,i] = transform.rotate(img_padding[...,i],
                                                  rotation, 
                                                  mode='edge', 
                                                  preserve_range =True)
        
    start_idx = int((pic_size_z-throughplane_depth)/2)

    img_padding = img_padding[...,start_idx:int(start_idx+throughplane_depth)]
    
    save_data(img_padding,
              save_path=save_path,
              name=save_name,
              pixdim=pixdim,
              ort=ort)
                
    return img_padding    
      
    