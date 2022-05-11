#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main code for AFFIRM-SVR pipeline

both deals with the simulated and also real-world data with NiftyMIC function embedded.

the code will be released step by step once the entire DL-based processing pipeline has been finished, 
including DL-based fetal brain extraction, motion detection and estimation, motion correction, and 
super-resolution reconstruction.

\author   Wen Allard Shi, JHU SOM BME
\date     02/2021

"""

import os
import sys
import shutil
import scipy
import numpy as np
import nibabel as nib
import multiprocessing as mp
from nipype.interfaces import fsl
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from dipy.denoise.nlmeans import nlmeans
#from dipy.denoise.noise_estimate import estimate_sigma
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

#from recon import ReconSolver
from util import save_data, CenterAlignment
from networks import AFFIRM
from registration import Slice2VolumeRegistration, Volume2VolumeRegistration
      
class fetalDL_pipeline(object):
    """
    will update in the next version
    
    """
    
    def __init__(self,
                 path_data,
                 moco_model_path,
                 recon_model_path,
                 ort,
                 nb_slice=24,
                 inplane_shape=(192,192),
                 pixdim_recon=0.8,
                 s_thickness=4.0,
                 init_sigma=1.0,
                 throughplane_depth=144,
                 nb_interleave=2,
                 nb_loop=3,
                 denoise=False,
                 simulation=False,
                 moco_model='AFFIRM',
                 recon_model='fetalRecon',
                 regis_type='FLIRT'):
    
        self.path_data = path_data
        self.moco_model_path = moco_model_path
        self.recon_model_path = recon_model_path    
        self.nb_stack = len(ort)
        self.nb_interleave = nb_interleave  
        self.st_ratio = s_thickness/pixdim_recon
        self.idx_slice = np.linspace(0,nb_slice-1,nb_slice)
        self.recurrent = 1
        self.nb_loop = nb_loop
        self.alpha = [0.02, 0.10, 0.25]
        self.sigma_model = 0.80
        self.vol_shape = (*inplane_shape, throughplane_depth)
        self.start_idx = int(int(throughplane_depth-nb_slice*self.st_ratio+1)/2)
        
        assert len(self.alpha) == self.nb_loop,  ValueError('nb of alpha ')    
   
    def _run_moco(self):
        pass
              
    def _run_recon(self):
        pass
            
    def _run(self):        
        pass



if __name__ == '__main__':
    
    dir_realdata = r'./Data/demo/'
    dir_data = dir_realdata
    dir_model =r'./Model/'
    dir_template = r'./Atlas/'
    
    ort = ['axi','cor','sag']
    prefix = ['AXI','COR','SAG']
    names = os.listdir(dir_data)    
    nb_subject = len(names)
    denoise = False
    simulation = False
    init_reference = True     # set Ture when first use 
    test = False
    reg_sigma = 1.0 if init_reference else 0
    # set outlier rejection
    final_threshold = 0.87 if init_reference else 0.93
    nb_cores = 1 if init_reference else int(mp.cpu_count())

    nb_slice = 24
    nb_stack = len(ort)
    nb_interleave = 2
    pixdim_recon = 0.8
    s_thickness = 4.0
    st_ratio = s_thickness/pixdim_recon
    throughplane_depth = 144
    inplane_shape = (192,192)
    idx_slice = np.linspace(0,nb_slice-1,nb_slice)
    recurrent = 4
    nb_loop = 3
    alpha = [0.02, 0.10, 0.25]
    sigma,sigma_init = 0.52, 0.80
    model_name = 'AFFIRM'
    vol_shape = (*inplane_shape, throughplane_depth)
    start_idx = int(int(throughplane_depth-nb_slice*st_ratio+1)/2)
    regis_type = 'FLIRT'
    
    
    assert len(alpha) == nb_loop,  ValueError('nb of alpha ')
    
    ###################################################################################

    print('='*30+' DL-fetalMoCoSRR '+'='*30)
    ph.print_info('\tNumber of Subject: {}'.format(nb_subject))
    ph.print_info('\tnumber of stack: {}'.format(nb_stack))
    ph.print_info('\torientation: {}'.format(ort))
    ph.print_info('\tnumber of interleave: {}'.format(nb_interleave))
    ph.print_info('\tnumber of slice: {}'.format(nb_interleave*nb_slice))
    ph.print_info('\tinplane-resolution: {} mm'.format(pixdim_recon))
    ph.print_info('\tinplane FOV: {}'.format(inplane_shape))
    ph.print_info('\tvolume registration type: {}'.format(regis_type))    
    ph.print_info('\tS2V initilization model: {}'.format(model_name))    
    ph.print_info('\tnumber of SVR-SRR loop: {}'.format(nb_loop))    
    ph.print_info('\tnumber of cpu core: {}'.format(nb_cores))
    
    print('='*77)

    # load network
    if test == False and init_reference: 
        
        if model_name == 'AFFIRM':
            model = AFFIRM(vol_shape=vol_shape,
                           nb_slice=nb_slice,
                           st_ratio=st_ratio,
                           nb_stack=nb_stack,
                           ort=ort,
                           sigma=sigma,
                           sigma_init=sigma_init,
                           recurrent=recurrent,
                           fusion_type='nonlocal',
                           predict=True)

            load_model_file = dir_model+model_name+'_iter4.h5'
        
        else:
            raise ValueError('model is not defined!')
            
            
    model.load_weights(load_model_file, by_name=True)
    
    # this part is done by dl-based center alignment, here we intially use manual adapatation    
    rotation_angle = [-60,-60,0]
                    
    for name in names:
        ph.print_info('\tStart preprocessing {}'.format(name))
    
        ga = int(np.round(int(name[:2])+int(name[3])/7)+1)
        if ga < 23: ga = 23
        if ga > 37: ga = 37
        
        if not os.path.isdir(dir_realdata+name+'/recon'):
            os.makedirs(dir_realdata+name+'/recon')
                                            
        for i in range(nb_stack):
            
            img_rawdata = nib.load(dir_realdata+name+'/T2_HASTE/2MM/GD/'+prefix[i]+'/'+ort[i]+'_0_biascorrect.nii.gz')  
            inplane_rs1 = img_rawdata.header['pixdim'][1]/pixdim_recon
            inplane_rs2 = img_rawdata.header['pixdim'][2]/pixdim_recon
            
            # ensure the orientation of the stack
#                img_ort = img_rawdata.hearder[]

            img_data = img_rawdata.get_data()
            
#                if denoise:
#                    sigma = estimate_sigma(img_data)
#                    img_data = nlmeans(img_data, sigma=sigma, patch_radius=1, block_radius=2, rician=True)
#                    
            mask_rawdata = nib.load(dir_realdata+name+'/T2_HASTE/2MM/GD/'+prefix[i]+'/'+ort[i]+'_0_mask.nii')
                
#                mask_rawdata = nib.load(dir_realdata+name+'/'+ort[i]+'_mask.nii')
            mask_data = mask_rawdata.get_data()
            img_data = np.multiply(img_data,mask_data)
            img_data[img_data<0] = 0
            img_data = scipy.ndimage.interpolation.zoom(img_data,(inplane_rs1,inplane_rs2,1),order=1)
                                            
            slices = CenterAlignment(img_rawdata=img_data,
                                     save_path=dir_realdata+name+'/recon/',
                                     save_name=ort[i],
                                     inplane_shape=inplane_shape,
                                     throughplane_depth=int(nb_slice*nb_interleave),
                                     ort=ort[i],
                                     pre_rot=0,
                                     rotation=rotation_angle[i],
                                     pixdim=[pixdim_recon,pixdim_recon,s_thickness/nb_interleave],
                                     interleave=1,
                                     init_idx=0)
             
#            reference = nib.load(dir_realdata + name + r'/recon/axi.nii.gz')
#            reference_data = reference.get_data()
#            
#            idx_sp = [int(2*i-1) for i in range(nb_slice)] 
#            
#            
#            save_data(reference_data[...,idx_sp],
#                      save_path=dir_realdata+name+'/recon/',
#                      name='reference_temp',
#                      pixdim=[pixdim_recon,pixdim_recon,4],
#                      ort='axi')
#                 
        # choose a baseline in atlas
        ## This should be combined with our automated motion detection algorithm, will be released once the related paper is published     
        ##########################################################################
        if True:
            dir_in = dir_realdata + name + r'/recon/axi.nii.gz'
            
        
        dir_temp = dir_template+'GA'+str(ga)+'.nii.gz'
        dir_out = dir_realdata + name + r'/recon/reference.nii.gz'                      
        ##########################################################################
        
        if regis_type == 'FLIRT':
            comfine = ' -searchrx -30 30 -searchry -30 30 -searchrz -30 30 '
            os.system('flirt -in '+dir_in+' -ref '+dir_temp+' -dof 6 '+'-out '+dir_out+comfine)            

        else:
            stack = sitk.ReadImage(dir_in, sitk.sitkFloat32)
            template = sitk.ReadImage(dir_temp, sitk.sitkFloat32)
            volume = Volume2VolumeRegistration(stack, template, dof=6)
            volume = sitk.Cast(volume, sitk.sitkFloat32)
            sitk.WriteImage(volume, dir_out)
    
#                flt.inputs.in_file = dir_in
#                flt.inputs.reference = dir_temp
#                flt.inputs.out_file = dir_out
#                flt.run()

        volume = nib.load(dir_out)
        volume = volume.get_data()

        volume = CenterAlignment(img_rawdata=volume,
                                 save_path=dir_realdata+name+'/recon/',
                                 save_name='reference',
                                 inplane_shape=inplane_shape,
                                 throughplane_depth=throughplane_depth,
                                 pixdim=[0.8]*3,
                                 interleave=1,
                                 automatic=False,
                                 init_idx=0)
        
        volume = volume[...,start_idx:int(start_idx+nb_slice*st_ratio)]
        volume = volume.reshape((1,*volume.shape,1))             
        volume = volume/np.max(volume)
                                          
        if not os.path.isdir(dir_realdata+name+'/recon/correct'):
            os.makedirs(dir_realdata+name+'/recon/correct') 

        # TODO optimization
        elif init_reference:
    
            shutil.rmtree(dir_realdata+name+'/recon/correct')
            os.makedirs(dir_realdata+name+'/recon/correct') 
        else:
            print('test again!')
            
        if os.path.exists(dir_realdata+name+'/recon/recon.nii.gz'):
            recon = nib.load(dir_realdata+name+'/recon/recon.nii.gz')
            recon_data = recon.get_data()
            recon_mask = nib.load(dir_realdata+name+'/recon/recon_mask.nii.gz')
            recon_mask_data = recon_mask.get_data()

            save_data(np.multiply(recon_data,recon_mask_data), 
                      save_path=dir_realdata+name+'/recon/', 
                      name='recon_reference', 
                      params=[0.,0.,0.,0.,0.,0.,], 
                      ort='axi', 
                      mask=False, 
                      pixdim=[0.8,0.8,0.8])
                            
        # TODO integrate with superresolution
        if init_reference:
            path_reference = dir_realdata+name+'/recon/reference.nii.gz'
            input_volume = np.clip(volume/np.max(volume),0,1)
            pred_transform = np.zeros((1,int(nb_slice*nb_stack*nb_interleave),6))
            
            for i in range(nb_interleave):
                idx = [int(nb_interleave*k+i) for k in range(nb_slice)]
                idx_pred = [int(nb_interleave*k+i) for k in range(int(nb_slice*nb_stack))]                
                input_slices = np.zeros((1,*inplane_shape,int(nb_slice*nb_stack),1))
                
                for j in range(nb_stack):
                    data_temp = nib.load(dir_realdata+name+'/recon/'+ort[j]+'.nii.gz')
                    data_temp = data_temp.get_data()
                    input_slices[0,...,int(nb_slice*j):int(nb_slice*(j+1)),0] = data_temp[...,idx]/np.max(data_temp[...,idx])
                    
                              
                input_slices = np.clip(input_slices,0,1)
                
                
                # *** modify the model input ***
                prediction = model.predict([input_slices, input_volume])
#                        prediction = model.predict([input_slices])
                pred_transform[:,idx_pred,:] = prediction.astype('float64')
            
            np.save(dir_realdata+name+'/recon/transform_pred.npy', pred_transform)
                                       
        else:
            path_reference = dir_realdata+name+'/recon/recon_reference.nii.gz'
            pred_transform = np.load(dir_realdata+name+'/recon/transform_pred.npy')

        # S2VRegistration
        for i in range(nb_stack):
            registration = Slice2VolumeRegistration(path_slices=dir_realdata+name+'/recon/'+ort[i]+'.nii.gz', 
                                                    path_volume=path_reference, 
                                                    ort=ort[i], 
                                                    init_transform=pred_transform[0,int(nb_slice*nb_interleave*i):int(nb_slice*nb_interleave*(i+1)),:],
                                                    savepath=dir_realdata+name+'/recon/correct/',
                                                    nb_slice=int(nb_slice*nb_interleave),
                                                    sigma=reg_sigma,
                                                    similarity_threshold=final_threshold,
                                                    nb_cores=nb_cores)
            registration._run()
   
           # for reconstruction part; refer to niftymic command
           # will release a self-recon demo in the next version
                
                # TODO: SRR reconstruction TV1L2
                # TODO: update DL-based model SRR
                
#                stacks = [None]*nb_stack
#                slices = []
#                recon0_sitk = sitk.ReadImage(dir_realdata+name+'/recon/reference.nii.gz', sitk.sitkFloat32)
#                recon0_itk = sitkh.get_itk_from_sitk_image(recon0_sitk)
#                  
#                for i in range(nb_stack):
#                   stacks[i] = sitk.ReadImage(dir_realdata+name+'/recon/'+ort[i]+'.nii.gz', sitk.sitkFloat32)
#                                
#                   for j in range(stacks[i].GetDepth()):
#                       tfm_path_temp = os.path.join(dir_realdata+name+'/recon/correct/','%s_slice%d.tfm'%(ort[i],j))
#                      
#                       if os.path.exists(tfm_path_temp):
#                           transform_slice_sitk = sitkh.read_transform_sitk(tfm_path_temp)              
#                           transform_stack_sitk_inv = sitk.Euler3DTransform()
#                           slices += [stacks[i][:,:,j:j+1]]   
#                                                                      
#                           transform_slice_sitk = sitkh.get_composite_sitk_affine_transform(transform_slice_sitk, transform_stack_sitk_inv)                            
#                           slice_affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(slices[-1])                                                 
#                           affine_transform = sitkh.get_composite_sitk_affine_transform(transform_slice_sitk, slice_affine_transform_sitk)
#                                            
#                           # reset the parameter
#                           origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(affine_transform, slices[-1])
#                           direction = sitkh.get_sitk_image_direction_from_sitk_affine_transform(affine_transform, slices[-1]) 
#                           slices[-1].SetOrigin(origin)
#                           slices[-1].SetDirection(direction)
#                
#                SRR0 = ReconSolver(slices=slices,
#                                   slice_spacing=np.array([pixdim_recon,pixdim_recon,s_thickness/nb_interleave]),
#                                   reconstruction_itk=recon0_itk,
#                                   reconstruction_sitk=recon0_sitk)
#                
#                print(' Start SRR ... ')
#                SRR0._run()
#                recon = SRR0.get_reconstruction()
#               
##                SDA = ScatteredDataApproximation(slices=slices,
##                                     reconstruction_sitk=recon0_sitk)    
##    
##                SDA._run()                
##                recon = SDA._get_sda()
##            
#                recon = sitk.Cast(recon, sitk.sitkFloat32)
#                sitk.WriteImage(recon,dir_realdata+name+'/recon/recon.nii.gz')
                            
