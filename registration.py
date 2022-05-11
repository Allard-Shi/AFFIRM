#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this file contains relevant utils for registration

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     02/2021

"""

import os
import abc
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import multiprocessing as mp
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
from util import motion_affine

class Registration(object):
    
    def __init__(self,
                 savepath,
                 learningrate=1.0):
                
        self.savepath = savepath
        self.learningrate = learningrate
    
    @abc.abstractmethod
    def _run(self):
        
        pass
        
#    @abc.abstractmethod
#    def _get_warped_object(self):
#        
#        pass
    
class Slice2VolumeRegistration(Registration):
    """    
    slice-to-volume registration    
    enable paralled computation (acceleration:*nb_core)
    
    """
    def __init__(self,
                 path_slices,
                 path_volume, 
                 ort, 
                 init_transform, 
                 savepath,
                 nb_slice,
                 pixrecon=0.8,
                 gt_path=None, 
                 similarity_threshold=0.85,
                 learningrate=1.0,
                 sigma=1.0,
                 debug=False,
                 nb_iter=1,
                 nb_cores=1,
                 parallel=True):
        
        self.path_slices = path_slices
        self.path_volume = path_volume
        self.ort = ort 
        self.init_transform=init_transform
        self.pixrecon = pixrecon
        self.gt_path =  gt_path
        self.similarity_threshold = similarity_threshold                 
        self.sigma = sigma
        self.debug = debug
        self.nb_iter = nb_iter
        self.nb_slice = nb_slice
        self.nb_cores = nb_cores if nb_cores >= nb_slice else 1
        self.parallel = False if self.nb_cores == 1  else True
        
        if self.parallel == False:
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetMetricAsCorrelation()
            registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1,
                                                                          numberOfIterations=100,
                                                                          lineSearchUpperLimit=2)
            
            registration_method.SetOptimizerScalesFromJacobian()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[3,2,1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1.5,1.0,0])
            self.registration_method = registration_method 
            self.slices = sitk.ReadImage(self.path_slices, sitk.sitkFloat32)
            self.volume = sitk.ReadImage(self.path_volume, sitk.sitkFloat32)

        super().__init__(savepath=savepath,
                         learningrate=learningrate)
    
                
#    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[3,3,3,0,0,0],
#                                                 stepLength=np.pi/60)
#    registration_method.SetOptimizerScales([1,1,1,0,0,0])
    
#    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.2,
#                                                                          minStep=1e-7,
#                                                                          numberOfIterations=400,
#                                                                          gradientMagnitudeTolerance=1e-9)
#    
#    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1,
#                                                                  minStep=1e-6,
#                                                                  numberOfIterations=1000,
#                                                                  gradientMagnitudeTolerance=1e-9)
        
          
    def _run(self):
        
        #    initial_transform = sitk.VersorRigid3DTransform()
                
        if self.sigma != 0 and self.parallel == False:
            gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
            gaussian.SetSigma(self.sigma)
            self.volume = gaussian.Execute(self.volume)
        
        ph.print_info('Slice-to-Volume Registration')
        ph.print_info('\tSelect Metric: %s' %('Correlation'))
        ph.print_info('\tLearning Rate: %f' %(self.learningrate))
        
        if self.parallel == False:
            ph.print_info('\tOptimizer\'s stopping condition: %s' % (
                self.registration_method.GetOptimizerStopConditionDescription()))
            ph.print_info('\tFinal metric value: %s' % (
                self.registration_method.GetMetricValue()))
    
        similarity = []
        reject_idx = []
    
        if self.parallel:
            
            ph.print_info('\tMulti-CPU core paralleled calculation')
            pool = mp.Pool(self.nb_cores)
            output = pool.map(self._run_registration, [s for s in range(self.nb_slice)])
            
            for i in range(self.nb_slice):
                similarity += output[i][0]
                reject_idx += output[i][1]
        else:
            for i in tqdm(range(self.slices.GetDepth())):
                similarity_temp, reject_idx_temp = self._run_registration(i)
                similarity += similarity_temp
                reject_idx += reject_idx_temp
        
        ph.print_info('Summary Slice2VolumeRegistration:')
        print('Similarity: {}'.format(similarity))
        print('Reject Slices: {}'.format(reject_idx))
            
    def _run_registration(self, i):
        
        if self.parallel:
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetMetricAsCorrelation()
            registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1,
                                                                              numberOfIterations=100,
                                                                              lineSearchUpperLimit=2)
            registration_method.SetOptimizerScalesFromJacobian()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[3,2,1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1.5,1,0])
            
            slices = sitk.ReadImage(self.path_slices, sitk.sitkFloat32)
            volume = sitk.ReadImage(self.path_volume, sitk.sitkFloat32)
            
            if self.sigma != 0:
                gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
                gaussian.SetSigma(self.sigma)
                volume = gaussian.Execute(volume)
            
        else:
            slices = self.slices
            volume = self.volume
            registration_method = self.registration_method
            
        fixed_slice = slices[:,:,i:i+1]
        fixed_slice_data = sitk.GetArrayFromImage(fixed_slice)        
        nb_nonzero_voxel = np.where(fixed_slice_data!=0)
        similarity = [0]
        path_transform = os.path.join(self.savepath,'%s_slice%d.tfm'%(self.ort,i))
        reject_idx = []
        
        if len(nb_nonzero_voxel[0]) > 0: 
                       
            if self.gt_path is not None:
                initial_transform_path = os.path.join(self.gt_path,'%s_slice%d.tfm'%(self.ort,i))
                initial_transform = sitk.ReadTransform(initial_transform_path)
                
            initial_transform = sitk.VersorRigid3DTransform()
                             
            if os.path.exists(path_transform):
                temp_sitk = sitk.AffineTransform(sitk.ReadTransform(path_transform))   
                initial_transform.SetMatrix(temp_sitk.GetMatrix())
                initial_transform.SetTranslation(temp_sitk.GetTranslation())
                
            else:
                
                motion_rot, motion_disp = motion_affine(self.init_transform[i,:], 
                                                        [0,0,0,0,0,0], 
                                                        pixrecon=self.pixrecon, 
                                                        ort='axi')
    
                motion_rot = np.ndarray.flatten(motion_rot)
                initial_transform.SetMatrix(motion_rot)
                initial_transform.SetTranslation(motion_disp)
            
            # slice-to-volume registration
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            registration_transform_sitk = registration_method.Execute(fixed_slice, volume)            
            
            similarity[-1] = abs(round(registration_method.GetMetricValue(),3))
            
            if self.debug:
                print('*'*40)
                print('similarity: ',similarity[-1])
                print('*'*40)
                       
            if np.abs(similarity[-1]) > self.similarity_threshold:                
                registration_transform_sitk = sitk.VersorRigid3DTransform(registration_transform_sitk)
                tmp = sitk.AffineTransform(3)        
                tmp.SetMatrix(registration_transform_sitk.GetMatrix())
                tmp.SetTranslation(registration_transform_sitk.GetTranslation())
                tmp.SetCenter(registration_transform_sitk.GetCenter())
                registration_transform_sitk = tmp
                
                if self.debug:
                    print('======= registration_transform_sitk ====== ')
                    sitkh.print_sitk_transform(registration_transform_sitk)
    
                sitk.WriteTransform(registration_transform_sitk, path_transform)
            else:
                
                if os.path.exists(path_transform):
                    os.remove(path_transform)
                reject_idx = [i]
        else:
            
            if os.path.exists(path_transform):
                os.remove(path_transform)
            reject_idx = [i]
                                
        return similarity, reject_idx


# TODO object
def Volume2VolumeRegistration(fixed_sitk,
                              moving_sitk,
                              init_transform=None,
                              dof=6):
    
    """    
    Volume2Volume Registrtion
    
    TODO: some bugs; update further 
    
    """
    

    registration_method = sitk.ImageRegistrationMethod()
    initial_transform = sitk.VersorRigid3DTransform()
    
    if init_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(fixed_sitk, 
                                                              moving_sitk,
                                                              initial_transform,
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetMetricAsCorrelation()

    registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1,
                                                                  numberOfIterations=100,
                                                                  lineSearchUpperLimit=2)
    
    registration_method.SetOptimizerScalesFromJacobian()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[3,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1.5,1,0])


    ph.print_info('Summary SimpleItkRegistration:')
    ph.print_info('\tOptimizer\'s stopping condition: %s' % (
        registration_method.GetOptimizerStopConditionDescription()))
    ph.print_info('\tFinal metric value: %s' % (
        registration_method.GetMetricValue()))
    ph.print_info('Volume-to-Volume Registration')
    
                # slice to volume registration
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_transform_sitk = registration_method.Execute(fixed_sitk, moving_sitk)
    similarity = registration_method.GetMetricValue()
    
    print('*'*40)
    print('similarity: ',similarity)
    print('*'*40)

    volume = sitk.Resample(fixed_sitk,
                           moving_sitk,
                           registration_transform_sitk,
                           sitk.sitkLinear,
                           0.,fixed_sitk.GetPixelIDValue())
    
    return volume

