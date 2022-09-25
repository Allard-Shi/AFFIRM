"""


This file is partly modified from the NiftyMIC source code .


author: Wen Shi 05/2021, JHU SOM & ZJU BME

"""
import os
import itk
import numpy as np
import SimpleITK as sitk
from linear_operators import LinearOperators
import nsol.linear_operators as linop
import nsol.tikhonov_linear_solver as tk
import pysitk.simple_itk_helper as sitkh


class ReconSolver(object):
    """
    
    
    
    """
    
    def __init__(self,
                 slices,
                 slice_spacing,
                 reconstruction_itk,
                 reconstruction_sitk,
                 alpha=0.035,
                 iter_max=5):
        
        self.nb_slices = len(slices)
        self.slices = slices
        self.slice_spacing = slice_spacing
        self.reconstruction_itk = reconstruction_itk
        self.reconstruction_sitk = reconstruction_sitk
        self.alpha = alpha
        self.iter_max = iter_max
        
        
        self._itk2np = itk.PyBuffer[itk.Image.D3]
        self._linear_operators = LinearOperators(deconvolution_mode='full_3D',
                                                 predefined_covariance=None,
                                                 alpha_cut=3,
                                                 image_type=itk.Image.D3)
        
        self.N_slice_voxels = np.array(self.slices[0].GetSize()).prod()
        self.N_total_slice_voxels = self.N_slice_voxels * len(self.slices)
        self.N_voxels_recon = np.array(reconstruction_sitk.GetSize()).prod()
        self.reconstruction_shape = sitk.GetArrayFromImage(self.reconstruction_sitk).shape
        self._x_scale = sitk.GetArrayFromImage(self.reconstruction_sitk).max()
        
        if self._x_scale == 0:
            self._x_scale = 1
        
        
    def _get_itk_image_from_array_vec(self, nda_vec, image_itk_ref):

        shape_nda = np.array(image_itk_ref.GetLargestPossibleRegion().GetSize())[::-1]
        image_itk = self._itk2np.GetImageFromArray(nda_vec.reshape(shape_nda))
        image_itk.SetOrigin(image_itk_ref.GetOrigin())
        image_itk.SetSpacing(image_itk_ref.GetSpacing())
        image_itk.SetDirection(image_itk_ref.GetDirection())

        return image_itk
    
    
    def _Mk_Ak(self, reconstruction_itk, slice_k):

        # Compute A_k x
        slice_k_itk = sitkh.get_itk_from_sitk_image(slice_k)
        Ak_reconstruction_itk = self._linear_operators.A_itk(reconstruction_itk, slice_k_itk, self.slice_spacing)


        return Ak_reconstruction_itk


    def _MA(self, reconstruction_nda_vec):

        # Convert reconstruction data array back to itk.Image object
        x_itk = self._get_itk_image_from_array_vec(reconstruction_nda_vec, self.reconstruction_itk)

        MA_x = np.zeros(self.N_total_slice_voxels)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, slice_i in enumerate(self.slices):

            i_max = i_min + self.N_slice_voxels
            # Compute M_k A_k y_k
            slice_itk = self._Mk_Ak(x_itk, slice_i)
            slice_nda = self._itk2np.GetArrayFromImage(slice_itk)

            # Fill corresponding elements
            MA_x[i_min:i_max] = slice_nda.flatten()
            i_min = i_max

        return MA_x
    
    
    def _Ak_adj_Mk(self, slice_itk):

        Mk_slice_itk = slice_itk

        # Compute A_k^* M_k y_k
        Mk_slice_itk = self._linear_operators.A_adj_itk(Mk_slice_itk, self.reconstruction_itk, self.slice_spacing)

        return Mk_slice_itk
    
    
    def _A_adj_M(self, stacked_slices_nda_vec):

        # Allocate memory
        A_adj_M_y = np.zeros(self.N_voxels_recon)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, slice_i in enumerate(self.slices):

            # Define index for last voxel to specify current slice
            i_max = i_min + self.N_slice_voxels

            # Extract 1D corresponding to current slice and convert it to
            # itk.Object
            slice_i_itk = sitkh.get_itk_from_sitk_image(slice_i)
            slice_itk = self._get_itk_image_from_array_vec(stacked_slices_nda_vec[i_min:i_max], slice_i_itk)

            # Apply A_k' M_k on current slice
            Ak_adj_Mk_slice_itk = self._Ak_adj_Mk(slice_itk)
            Ak_adj_Mk_slice_nda_vec = self._itk2np.GetArrayFromImage(Ak_adj_Mk_slice_itk).flatten()

            # Add contribution
            A_adj_M_y += Ak_adj_Mk_slice_nda_vec

            i_min = i_max

        return A_adj_M_y

    def _get_M_y(self):

        My = np.zeros(self.N_total_slice_voxels)
        
        i_min = 0

        for i, slice_i in enumerate(self.slices):

            # Define index for last voxel to specify current slice
            i_max = i_min + self.N_slice_voxels

            slice_itk = sitkh.get_itk_from_sitk_image(slice_i)
            slice_nda_vec = self._itk2np.GetArrayFromImage(slice_itk).flatten()

            # Fill respective elements
            
            My[i_min:i_max] = slice_nda_vec
            i_min = i_max

        return My
    
    def get_solver(self):
        
        A = lambda x: self._MA(x)
        A_adj = lambda x: self._A_adj_M(x)
        b = self._get_M_y()
        x0 = sitk.GetArrayFromImage(self.reconstruction_sitk).flatten()
        
        spacing = np.array(self.reconstruction_sitk.GetSpacing())
        linear_operators = linop.LinearOperators3D(spacing=spacing)
        grad, grad_adj = linear_operators.get_gradient_operators()

        X_shape = self.reconstruction_shape
        Z_shape = grad(x0.reshape(*X_shape)).shape

        B = lambda x: grad(x.reshape(*X_shape)).flatten()
        B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        # Set up solver
        solver = tk.TikhonovLinearSolver(A=A,
                                         A_adj=A_adj,
                                         B=B,
                                         B_adj=B_adj,
                                         b=b,
                                         x0=x0,
                                         x_scale=self._x_scale,
                                         alpha=self.alpha,
                                         data_loss='linear',
                                         data_loss_scale=1,
                                         verbose=1,
                                         minimizer='lsmr',
                                         iter_max=self.iter_max,
                                         bounds=(0, np.inf))

        return solver
    
    def _run(self):

        solver = self.get_solver()

#        self._print_info_text()

        # Run reconstruction
        solver.run()

        # After reconstruction: Update member attribute
        self.reconstruction_itk = self._get_itk_image_from_array_vec(solver.get_x(), self.reconstruction_itk)
        self.reconstruction_sitk = sitkh.get_sitk_from_itk_image(self.reconstruction_itk)

    def get_reconstruction(self):
         
        return self.reconstruction_sitk

    def get_inital_volume(self):
        
        x0 = self._get_M_y()
        
        init_vol = self._A_adj_M(x0)   
        
        init_vol = self._get_itk_image_from_array_vec(init_vol, self.reconstruction_itk)
        init_vol_sitk = sitkh.get_sitk_from_itk_image(init_vol)
                
#        xx = self._MA(sitk.GetArrayFromImage(init_vol_sitk))
        
        return init_vol_sitk


class ScatteredDataApproximation(object):
    """
    scattered data interpolation 
    
    
    
    
    """
        
    def __init__(self, slices, reconstruction_sitk, sigma=.8):
        self.slices = slices
        self.reconstruction_sitk = reconstruction_sitk
        self.N_slices = len(slices)
        self.sigma = sigma
        
    def _run(self):
        
        shape = sitk.GetArrayFromImage(self.reconstruction_sitk).shape
        helper_N_nda = np.zeros(shape)
        helper_D_nda = np.zeros(shape)

        default_pixel_value = 0.0

        for i in range(self.N_slices):

            slice_sitk = self.slices[i]
            slice_sitk.SetSpacing([0.8,0.8,2])
            slice_sitk += 1
            
            slice_resampled_sitk = sitk.Resample(slice_sitk,
                                                 self.reconstruction_sitk,
                                                 sitk.Euler3DTransform(),
                                                 sitk.sitkNearestNeighbor,
                                                 default_pixel_value,
                                                 self.reconstruction_sitk.GetPixelIDValue())

            nda_slice = sitk.GetArrayFromImage(slice_resampled_sitk)
            ind_nonzero = nda_slice > 0
            helper_N_nda[ind_nonzero] += nda_slice[ind_nonzero] - 1
            helper_D_nda[ind_nonzero] += 1

        helper_D_nda[helper_D_nda == 0] = 1    
        nda = helper_N_nda/ helper_D_nda
        
        HR_volume_update = sitk.GetImageFromArray(nda)
        HR_volume_update.CopyInformation(self.reconstruction_sitk)

        self.sda_volume = HR_volume_update

    def _get_sda(self):
                
        return self.sda_volume
        

if __name__ == '__main__':

    path = r'./data/recon/test/1/'    
    path_correct = path + r'mc_gt/'
    
    recon0_sitk = sitk.ReadImage(path+'recon_gt.nii.gz', sitk.sitkFloat64)
    recon0_itk = sitkh.get_itk_from_sitk_image(recon0_sitk)
    
    print(recon0_sitk.GetSize())
    nb_slice = 24
    ort = ['axi','cor']
    nb_stack = len(ort)
    pixdim_recon = 0.8
    s_thickness = 4.0
    st_ratio = s_thickness/pixdim_recon
    inplane_shape = (192,192)
    volshape = (192,192,144)
    mask = False
    stacks = [None]*nb_stack
    slices = []
    
    for i, name in enumerate(ort):
       stacks[i] = sitk.ReadImage(path+name+'.nii.gz', sitk.sitkFloat32)
       
       for j in range(stacks[i].GetDepth()):
           tfm_path_temp = os.path.join(path_correct,'%s_slice%d.tfm'%(name,j))
          
           if os.path.exists(tfm_path_temp):
               transform_slice_sitk = sitkh.read_transform_sitk(tfm_path_temp)              
               transform_stack_sitk_inv = sitk.Euler3DTransform()
               slices += [stacks[i][:,:,j:j+1]]   
               
               if mask:
                  temp = sitk.GetArrayFromImage(slices[-1])
                  temp[:] = 1
                  
                  temp_sitk = sitk.GetImageFromArray(temp)
                  temp_sitk.SetDirection(slices[-1].GetDirection())
                  temp_sitk.SetSpacing(slices[-1].GetSpacing())
                  temp_sitk.SetOrigin(slices[-1].GetOrigin())
                  slices[-1] = temp_sitk
                                 
               transform_slice_sitk = sitkh.get_composite_sitk_affine_transform(transform_slice_sitk, transform_stack_sitk_inv)
                
               slice_affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(slices[-1])
                                     
               affine_transform = sitkh.get_composite_sitk_affine_transform(transform_slice_sitk, slice_affine_transform_sitk)
                                     
               origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(affine_transform, slices[-1])
               direction = sitkh.get_sitk_image_direction_from_sitk_affine_transform(affine_transform, slices[-1]) 
               slices[-1].SetOrigin(origin)
               slices[-1].SetDirection(direction)
    
    SRR0 = ReconSolver(slices=slices,
                       slice_spacing = np.array([pixdim_recon,pixdim_recon,s_thickness]),
                       reconstruction_itk=recon0_itk,
                       reconstruction_sitk=recon0_sitk)
    
    
#    recon = SRR0.get_inital_volume()

#    SRR0._run()
##    
#    recon = SRR0.get_reconstruction()
#    sitk.WriteImage(recon,path+'recon_niftymic.nii.gz')  
    
    SDA = ScatteredDataApproximation(slices=slices,
                                     reconstruction_sitk=recon0_sitk)    
    
    SDA._run()
    
    recon = SDA._get_sda()
    recon = sitk.Cast(recon, sitk.sitkFloat32)
    
    sitk.WriteImage(recon,path+'recon_sda.nii.gz')


