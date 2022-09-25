"""

This file contains MotionGenerator and DataGenerator for simulation and the network training

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020

"""

import sys
import random
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import InterpolatedUnivariateSpline
sys.path.append('./ext/neuron')
sys.path.append('./ext/pynd-lib')
sys.path.append('./ext/pytools-lib')

#import layers
from util import rotate_matrix, matrix2vec

## TODO: update to Quaternion 
class MotionGenerator(object):
    """
    Motion Generator (currently only for 3D volume)
    
    """
        
    def __init__(self, 
                 nb_slice, 
                 st_ratio, 
                 c_point=[3], 
                 motion_type='Euler',
                 g_angle=45, 
                 g_disp_xy=2.5, 
                 g_disp_z=2.5, 
                 l_angle=8, 
                 l_disp_xy=3.75, 
                 l_disp_z=3.75,                  
                 trs_angle=55,
                 trs_disp_xy=7.5, 
                 trs_disp_z=7.5,  
                 transform_params=None,
                 choose_psf=False,
                 add_noise=False,
                 dev_noise=0.03):      
        """
        
        Argument:
            nb_slice: num of slices
            st_ratio: slice thickness/recon isotropic resolution
            c_point: control point setting
            motion_type: rigid motion parameterization type, currently only support 'Euler'
            g_angle: global rotation offset range (degree)
            g_disp_xy: global in-plane displacement offset range (unit: voxel)
            g_disp_z: global through-plane displacement offset range (unit:voxel)
            l_angle: local rotation offset range (degree)
            l_disp_xy: local in-plane displacement offset range (unit: voxel)
            l_disp_z: local through-plane displacement offset range (unit:voxel)
            trs_angle: threshold of the maximum rotation angle (degree)
            trs_disp_xy: threshold of the maximum in-plane displacement
            trs_disp_z: threshold of the maximum through-plane displacement
            transform_params: customized transform params
            choose_psf: use default point-spread-function
            add_noise: add noise to the simulated data
            dev_noise: noise standard deviation
                        
        """

        self.nb_slice = nb_slice
        self.st_ratio = st_ratio
        self.c_point = c_point
        self.motion_type = motion_type
        self.g_angle = g_angle
        self.g_disp_xy = g_disp_xy
        self.g_disp_z = g_disp_z
        self.l_angle = l_angle
        self.l_disp_xy = l_disp_xy
        self.l_disp_z = l_disp_z
        self.trs_angle = trs_angle
        self.trs_disp_xy = trs_disp_xy
        self.trs_disp_z = trs_disp_z
        self.transform_params = None
        self.choose_psf = choose_psf
        self.add_noise = add_noise
        self.dev_noise = dev_noise
 
        assert self.motion_type in ['Euler', 'Quaternion'], ValueError('Motion type selection is not valid!')
    
    def _set_angle(self, scope_angle=15):
        """
        generate the trajectory of angle       
        scope_angle: control the average angle/sec        
        maximum averaging changing 8/(2*0.8) = 5 degrees/sec
        
        notice the relationship between nb_slice and c_point
        
        """
        
        idx = [0]
        angle = [0]        
        c_point = random.choice(self.c_point)
        
        while True:        
            step = np.random.randint(2, c_point+1)
            temp = idx[-1]+ step        
            if  temp < self.nb_slice-step:
                idx += [temp]
                angle += [angle[-1]+np.random.uniform(-scope_angle, scope_angle)]        
            else:break
        
        idx += [self.nb_slice-1]
        angle += [angle[-1]+np.random.uniform(-scope_angle, scope_angle)]
        angle -= np.mean(angle)
           
        return idx, angle
        
    def _set_disp(self, scope_disp):
        """
        generate the trajectory of displacement
        
        
        """
        idx = [0]
        disp = [0]
        c_point = random.choice(self.c_point)
        
        while True:
            step = np.random.randint(2, c_point+1)
            temp = idx[-1]+ step        
            if  temp < self.nb_slice-step:
                idx += [temp]
                disp += [disp[-1]+np.random.uniform(-scope_disp, scope_disp)]
            else:break
        
        idx += [self.nb_slice-1]
        disp += [disp[-1]+np.random.uniform(-scope_disp, scope_disp)]
        disp -= np.mean(disp)
        
        return idx, disp
    
    def _sampling_psf(self, scale=1.0):
        """
        in-plane point spread function (PSF) can be either a Gaussian or a sinc function 
        through-plane PSF can be a Gaussian function 
        
        we initally consider through-plane sampling as a truncated Gaussian function exp*(-dz^2/(2*sigma_z^2))    
        to accelerate the computation we choose window_size = 1*st_ratio
        the in-plane psf is not considered in this version
         
        """
        
        win = int(scale*self.st_ratio)
        sigma = self.st_ratio/2.3548        
        scale = np.arange(-(win-1)/2, (win+1)/2)
        
        kernel = np.exp(-np.multiply(scale,scale)/(2*sigma*sigma))    
        kernel = kernel/np.sum(kernel)
        
        return kernel

    def _get_motion(self):
         
        """
        the motion trajectory generator for six rigid transformation parameters in Euler representation 
            
        default TR = 800ms
            
        TODO: update other motion representation
        """
        
        if self.transform_params is not None:
            # TODO: validate transform_init
            assert self.transform_params.shape[-1] == 6, \
                ValueError('transform_params should have six params in Euler representation!')
                
            assert self.transform_params.shape[-2] == self.nb_slice, \
                ValueError('nb_slice is not equal to the given transform_params size! Got %d , should be %d' % ((self.transform_params.shape[-2], self.nb_slice)))
    
            transform = self.transform_params
                    
        else:  
            idx_slice = np.linspace(0, self.nb_slice-1, self.nb_slice)  
            
            global_rotation = np.random.uniform(-self.g_angle, self.g_angle, 3)
            global_displacement_xy = np.random.uniform(-self.g_disp_xy, self.g_disp_xy, 2)    
            global_displacement_z = np.random.uniform(-self.g_disp_z, self.g_disp_z, 1)       
            global_displacement = np.concatenate((global_displacement_xy, global_displacement_z))
            
            ax, ay, az, dx, dy, dz = [1000], [1000], [1000], [1000], [1000], [1000]
            while np.any(np.abs(ax+global_rotation[0]) >= self.trs_angle): idx_ax, ax = self._set_angle(self.l_angle)
            while np.any(np.abs(ay+global_rotation[1]) >= self.trs_angle): idx_ay, ay = self._set_angle(self.l_angle)
            while np.any(np.abs(az+global_rotation[2]) >= self.trs_angle) : idx_az, az = self._set_angle(self.l_angle)
            while np.any(np.abs(dx+global_displacement[0]) >= self.trs_disp_xy) : idx_dx, dx = self._set_disp(self.l_disp_xy)
            while np.any(np.abs(dy+global_displacement[1]) >= self.trs_disp_xy) : idx_dy, dy = self._set_disp(self.l_disp_xy)
            while np.any(np.abs(dz+global_displacement[2]) >= self.trs_disp_z) : idx_dz, dz = self._set_disp(self.l_disp_z)
            
            # Spline interpolator
            f_x_angle = InterpolatedUnivariateSpline(idx_ax, ax)
            f_y_angle = InterpolatedUnivariateSpline(idx_ay, ay)
            f_z_angle = InterpolatedUnivariateSpline(idx_az, az)    
            f_x_disp = InterpolatedUnivariateSpline(idx_dx, dx)
            f_y_disp = InterpolatedUnivariateSpline(idx_dy, dy)
            f_z_disp = InterpolatedUnivariateSpline(idx_dz, dz)
                            
            x_disp = f_x_disp(idx_slice).reshape((1,-1,1))+global_displacement[0]
            y_disp = f_y_disp(idx_slice).reshape((1,-1,1))+global_displacement[1]
            z_disp = f_z_disp(idx_slice).reshape((1,-1,1))+global_displacement[2]
        
            x_angle = f_x_angle(idx_slice).reshape((1,-1,1))+global_rotation[0]
            y_angle = f_y_angle(idx_slice).reshape((1,-1,1))+global_rotation[1]
            z_angle = f_z_angle(idx_slice).reshape((1,-1,1))+global_rotation[2]
                
            displacement = np.concatenate((x_disp,y_disp),axis=-1)
            displacement = np.concatenate((displacement,z_disp),axis=-1)
        
            rotation = np.concatenate((x_angle,y_angle),axis=-1)
            rotation = np.concatenate((rotation, z_angle),axis=-1)
                
            # generate images according to the motion trajectory
            transform = np.concatenate((displacement,rotation),axis=-1)
                
        return transform

    def _get_motion_corrupted_slice(self, 
                                    volume, 
                                    rigid_transform, 
                                    vol_shape, 
                                    scale=1.0,
                                    ort='axi', 
                                    anc_point=False):
        """
        Simulate inter-slice motion-corrupted images 
        
        in Hou's 2018 TMI paper, it can be observed that the sampling orientation scheme has  
        small effect on the prediction accuracy, if given adequate data. 
        
        Note: the order of xyz in itk format may be a little bit different 
        """
                             
        affine = sitk.AffineTransform(3)
        
        psf = self._sampling_psf(scale=scale) if self.choose_psf else [1]  
        w_size = len(psf)
            
        x0, y0, z0 = (vol_shape[0]-1)/2, (vol_shape[1]-1)/2, ((self.nb_slice-1)*self.st_ratio)/2
            
        # initialize the data in sitk
        fetus_data_temp = volume[0,...,0]
        fetus_data_temp = fetus_data_temp.transpose((2,1,0))
        data = sitk.GetImageFromArray(fetus_data_temp)
        data.SetOrigin([-(vol_shape[0]-1)/2,-(vol_shape[1]-1)/2,-(vol_shape[2]-1)/2])    
        data.SetSpacing([1,1,1])
        data.SetDirection(np.eye(3).flatten())
           
        corrupted_slice = np.zeros((1,*vol_shape[:2],self.nb_slice,1))
        idx = int((vol_shape[2]-(self.nb_slice-1)*self.st_ratio+1)-w_size)/2
        
    #    print('idx:',idx)
        if idx < 0 or idx >= vol_shape[2]:
            raise ValueError('st_ratio set too large! Got idx equals to {}.'.format(idx))
        
        for i in range(self.nb_slice):
            
            # rigid trans params
            motion_rot = rotate_matrix(rigid_transform[0,i,3:], ort=ort)[:3,:3]               
            motion_disp = rigid_transform[0,i,:3]
            affine.SetMatrix(motion_rot.ravel())
            affine.SetTranslation(motion_disp)
            mt_stack = sitk.Resample(data,data,affine,sitk.sitkLinear,0,data.GetPixelIDValue())
            idx_slice = int(idx+i*self.st_ratio)
            slab = sitk.GetArrayFromImage(mt_stack)
            slab = slab.transpose((2,1,0))
            corrupted_slice[0,...,i,0] = np.sum(np.multiply(slab[...,idx_slice:int(idx_slice+w_size)],psf),axis=-1)
            
            if self.add_noise:
                corrupted_slice += np.random.normal(0,self.dev_noise,corrupted_slice.shape)
                
        if anc_point:
            ap_1 = np.zeros((1,self.nb_slice,3))
            ap_2 = np.zeros((1,self.nb_slice,3))
            ap_3 = np.zeros((1,self.nb_slice,3))
            
            # Note: need to adapt to displacement
            for i in range(self.nb_slice):
                
                # not consider displacement since the training is very difficult for anchor point-based method
                motion_rot = rotate_matrix(rigid_transform[0,i,3:], ort=ort)[:3,:3] 
                ap_1[0,i,:] = np.dot(motion_rot,np.array([-x0,-y0,-z0+i*self.st_ratio])) + rigid_transform[0,i,:3]
                ap_2[0,i,:] = np.dot(motion_rot,np.array([0,0,-z0+i*self.st_ratio])) + rigid_transform[0,i,:3]
                ap_3[0,i,:] = np.dot(motion_rot,np.array([x0,-y0,-z0+i*self.st_ratio])) + rigid_transform[0,i,:3]
            
            return [corrupted_slice,ap_1,ap_2,ap_3]
                
        else:        
            return [corrupted_slice]

    @staticmethod
    def traj2rlt(rigid_transform, order='pos'):
        """
        convert to relative trajectory
    
    
        """
        nb_slice = rigid_transform.shape[1]    
        rlt_rigid_transform = np.zeros(rigid_transform.shape)
                
        if order == 'pos':
            for i in range(nb_slice-1):
                ant_matrix = rotate_matrix(rigid_transform[0,i,3:], ort='axi')[:3,:3]    
                post_matrix = rotate_matrix(rigid_transform[0,i+1,3:], ort='axi')[:3,:3]   
                matrix_temp = np.matmul(post_matrix,np.linalg.inv(ant_matrix))
                rlt_rigid_transform[0,i+1,3:] = matrix2vec(matrix_temp)[0,3:]
                rlt_rigid_transform[0,i+1,:3] = rigid_transform[0,i+1,:3] - rigid_transform[0,i,:3] 
                            
        else:        
            for i in range(nb_slice-1):
                ant_matrix = rotate_matrix(rigid_transform[0,i+1,3:], ort='axi')[:3,:3]    
                post_matrix = rotate_matrix(rigid_transform[0,i,3:], ort='axi')[:3,:3]   
                matrix_temp = np.matmul(post_matrix,np.linalg.inv(ant_matrix))
                rlt_rigid_transform[0,i,3:] = matrix2vec(matrix_temp)[0,3:]
                rlt_rigid_transform[0,i,:3] = rigid_transform[0,i,:3] - rigid_transform[0,i+1,:3] 
                   
        return rlt_rigid_transform

def data_generator(fetus_data, 
                   vol_shape, 
                   nb_slice, 
                   st_ratio, 
                   model,
                   nb_stack=1, 
                   ort=['axi'], 
                   batch_size=1, 
                   abs_loc=True,
                   pad_shape=None, 
                   area_threshold=400):
    '''
    Argument: 
        fetus_data: (#N,x,y,z,1)     
        vol_shape: volumetric size = (192,192,144)
        nb_slice: num of slices, e.g., 24
        st_ratio: slice thickness ratio = slice thickness/in-plane resolution e.g., 4mm/0.8mm = 5
        model: training model name
        nb_stack: num of stacks
        ort: orientation of the corresponding stack
        batch_size: batch size used in training session
        abs_loc: use absolute location
        pad_shape: padding shape
        area_threshold: the minimum informative voxel size
    
    '''
    subject_number = fetus_data.shape[0]
    
    assert len(ort) == nb_stack, 'each stack should be set orientation!'
#    displacement = np.zeros((batch_size,int(nb_slice*nb_stack),3))
    
    motion_generator = MotionGenerator(nb_slice=nb_slice, 
                                       st_ratio=st_ratio,
                                       choose_psf=True)
    
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()

    while True:

        # randomly select subject from the dataset
        idx = np.random.randint(subject_number, size=batch_size)
        sigma = np.random.uniform(0.5,1.0,1)[0]
                
        if model == 'HouNet2018':
            anc_point = True 
            ap_1 = np.zeros((1,int(nb_slice*nb_stack),3))
            ap_2 = np.zeros((1,int(nb_slice*nb_stack),3))
            ap_3 = np.zeros((1,int(nb_slice*nb_stack),3))
            
        else:
            anc_point = False
        
        rigid_transform = np.zeros((batch_size,int(nb_slice*nb_stack),6))
        input_slices = np.zeros((batch_size,*vol_shape[:2],int(nb_slice*nb_stack),1))
            
        for i in range(nb_stack):                
            rigid_transform[...,int(i*nb_slice):int((i+1)*nb_slice),:] = motion_generator._get_motion()                                   
                            
        data_fetus = np.ndarray((0,*vol_shape[:2], int(nb_slice*st_ratio),1))        
        start_idx = int(int(vol_shape[2]-nb_slice*st_ratio+1)/2)
#        print('start_idx: ',start_idx)      
        offset_z = np.zeros((batch_size,nb_slice))

        for i in range(batch_size):
            
            data_simulate = np.ndarray((1,*vol_shape[:2],0,1))
            idx_batch = idx[i]
            
            for j in range(nb_stack):
                
                data_simulate_temp = motion_generator._get_motion_corrupted_slice(volume=fetus_data[idx_batch:idx_batch+1,...],
                                                                                  rigid_transform=rigid_transform[...,int(nb_slice*j):int(nb_slice*(j+1)),:],
                                                                                  vol_shape=vol_shape, 
                                                                                  ort=ort[j], 
                                                                                  anc_point=anc_point)
                               
                data_simulate = np.concatenate((data_simulate, data_simulate_temp[0]),axis=-2)
                                                               
                if model == 'HouNet2018':
                    ap_1[i:i+1,int(nb_slice*j):int(nb_slice*(j+1)),:] = data_simulate_temp[1]
                    ap_2[i:i+1,int(nb_slice*j):int(nb_slice*(j+1)),:] = data_simulate_temp[2]
                    ap_3[i:i+1,int(nb_slice*j):int(nb_slice*(j+1)),:] = data_simulate_temp[3]
            
            input_slices[i:i+1,...] = data_simulate
            
            # idx validslice is used for outlier rejection
            idx_validslice = [k for k in range(int(nb_stack*nb_slice)) if len(np.where(data_simulate[0,...,k,0] != 0)[0]) > area_threshold]                  
            idx_validslice = np.array(idx_validslice).astype('int32')
                     
            data_fetus = np.concatenate((data_fetus,fetus_data[idx_batch:idx_batch+1,...,start_idx:int(start_idx+nb_slice*st_ratio),:]),axis=0)
              
            offset_z[i,...] = np.arange(start_idx,start_idx+nb_slice*st_ratio, st_ratio)
         
        # whether to use absolute location or relative location
        if not abs_loc: 
        
            rlt_rigid_transform = np.zeros(rigid_transform.shape)
            
            for i in range(nb_stack):
                rlt_rigid_transform[:,int(i*nb_slice):int((i+1)*nb_slice),:] = \
                MotionGenerator.traj2rlt(rigid_transform[:,int(i*nb_slice):int((i+1)*nb_slice),:])
                
            rotation = rlt_rigid_transform[...,3:]
        
        else:     
            rotation = rigid_transform[...,3:]
            displacement = rigid_transform[...,:3]

            vol_reference = np.zeros(data_fetus.shape)
            data_fetus_temp = sitk.GetImageFromArray(data_fetus[0,...,0])
            gaussian.SetSigma(sigma)
            data_fetus_temp = gaussian.Execute(data_fetus_temp)
            vol_reference[0,...,0] = sitk.GetArrayFromImage(data_fetus_temp)
            
        if model == 'AFFIRM':            
            
            # only run for batch size=1 currently     
            # TODO: volume calibration
            
#            vol_input = np.zeros((1,*vol_shape[:2],int(st_ratio*nb_slice),1))
#            rigid_transform = rigid_transform[:,list(idx_validslice),:]   
            rotation = rotation[:,list(idx_validslice),:] 
            displacement = displacement[:,list(idx_validslice),:] 
            
#            geodesic = geodesic[:,list(idx_validslice),:]
            rigid_transform_matrix = np.zeros((1,int(nb_slice*nb_stack),3, 4))
            rigid_transform_matrix = rigid_transform_matrix[:,list(idx_validslice),...] 
            rigid_transform_temp = rigid_transform[:,list(idx_validslice),:] 
                        
            for k in range(rigid_transform_matrix.shape[1]):
                rigid_transform_matrix[0,k,:,:3] = rotate_matrix(rigid_transform_temp[0,k,3:], ort='axi')[:3,:3]    
                rigid_transform_matrix[0,k,:,3] = rigid_transform_temp[0,k,:3]
            
            rigid_transform_matrix = rigid_transform_matrix.reshape((1,-1, 12))  
            idx_validslice = idx_validslice.reshape((batch_size,*idx_validslice.shape,1))
            invols  = [input_slices, rigid_transform, idx_validslice, data_fetus]                                               
            outvols = [rotation, displacement]
            
            yield (invols, outvols)  
                                              
        else:
            raise ValueError('Model name is not valid! {}'.format(model))
            
         
if __name__ == '__main__':  
    pass