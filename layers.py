"""
The customized network layers 

Some sources were modified from voxelmorph

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020
"""

import sys
sys.path.append('./ext/neuron')
sys.path.append('./ext/pynd-lib')
import math
import numpy as np
import neuron as ne
import tensorflow as tf
import keras 
import tensorflow.keras.backend as K
from keras.layers import Layer
from keras import (initializers, regularizers, constraints)
from keras.engine.topology import InputSpec
from utils import (is_affine, extract_affine_ndims, affine_shift_to_identity, affine_identity_to_shift, gaussian_blur)
import neuron.utils as nrn_utils

# make the following neuron layers directly available from vxm
SpatialTransformer = ne.layers.SpatialTransformer
LocalParam = ne.layers.LocalParam

def gelu(x):
    
    return 0.5*x*(1+tf.tanh(tf.sqrt(2/math.pi)*(x+0.044715*tf.pow(x,3))))


class Rescale(Layer):
    """ 
    Rescales a layer by some factor.
    """

    def __init__(self, scale_factor, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return x * self.scale_factor

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleTransform(Layer):
    """ 
    Rescales a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, zoom_factor, interp_method='linear',z_fix=False, **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.z_fix = z_fix
        super().__init__(**kwargs)

    def build(self, input_shape):

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('RescaleTransform must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        self.is_affine = is_affine(input_shape[1:])
        self.ndims = extract_affine_ndims(input_shape[1:]) if self.is_affine else int(input_shape[-1])

        super().build(input_shape)

    def call(self, inputs):

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            trf = inputs[0]
        else:
            trf = inputs

        if self.is_affine:
            return tf.map_fn(self._single_affine_rescale, trf, dtype=tf.float32)
        else:
            if self.zoom_factor[0] < 1:
                # resize
                trf = ne.layers.Resize(self.zoom_factor, name=self.name + '_resize')(trf)
                return Rescale(self.zoom_factor[0], name=self.name + '_rescale')(trf)
            else:
                # multiply first to save memory (multiply in smaller space)
                trf = Rescale(self.zoom_factor[0], name=self.name + '_rescale')(trf)
                return ne.layers.Resize(self.zoom_factor, name=self.name + '_resize')(trf)

    def _single_affine_rescale(self, trf):
        matrix = affine_shift_to_identity(trf)
        scaled_translation = tf.expand_dims(matrix[:, -1] * self.zoom_factor, 1)
        scaled_matrix = tf.concat([matrix[:, :-1], scaled_translation], 1)
        return affine_identity_to_shift(scaled_matrix)

    def compute_output_shape(self, input_shape):
        if self.is_affine:
            return (input_shape[0], self.ndims * (self.ndims + 1))
        else:
            output_shape = []
#            output_shape = [np.int(dim * self.zoom_factor) for dim in input_shape[1:-1]]
            for a,f in enumerate(input_shape[1:-1]):

                output_shape += [int(f * self.zoom_factor[a])]
                        
            return (input_shape[0], *output_shape, self.ndims)


class ComposeTransform(Layer):
    """ 
    Composes two transforms specified by their displacements. Affine transforms
    can also be provided. If only affines are provided, the returned transform
    is an affine, otherwise it will return a displacement field.

    We have two transforms:

    A --> B (so field/result is in the space of B)
    B --> C (so field/result is in the space of C)
    
    This layer composes a new transform.

    A --> C (so field/result is in the space of C)
    """

    def build(self, input_shape, **kwargs):

        if len(input_shape) != 2:
            raise Exception('ComposeTransform must be called on a input list of length 2.')

        # figure out if any affines were provided
        self.input_1_is_affine = is_affine(input_shape[0][1:])
        self.input_2_is_affine = is_affine(input_shape[1][1:])
        self.return_affine = self.input_1_is_affine and self.input_2_is_affine

        if self.return_affine:
            # extract dimension information from affine
            shape = input_shape[0][1:]
            if len(shape) == 1:
                # if vector, just compute ndims since length = N * (N + 1)
                self.ndims = int((np.sqrt(4 * int(shape[0]) + 1) - 1) / 2)
            else:
                self.ndims = int(shape[0])
        else:
            # extract shape information whichever is the dense transform
            dense_idx = 1 if self.input_1_is_affine else 0
            self.ndims = input_shape[dense_idx][-1]
            self.volshape = input_shape[dense_idx][1:-1]

        super().build(input_shape)

    def call(self, inputs):
        """
        Parameters
            inputs: list with two dense deformations
        """
        assert len(inputs) == 2, 'inputs has to be len 2, found: %d' % len(inputs)

        input_1 = inputs[0]
        input_2 = inputs[1]

        if self.return_affine:
            return tf.map_fn(self._single_affine_compose, [input_1, input_2], dtype=tf.float32)
        else:
            # if necessary, convert affine to dense transform
            if self.input_1_is_affine:
                input_1 = AffineToDense(self.volshape)(input_1)
            elif self.input_2_is_affine:
                input_2 = AffineToDense(self.volshape)(input_2)

            # dense composition
            return tf.map_fn(self._single_dense_compose, [input_1, input_2], dtype=tf.float32)

    def _single_dense_compose(self, inputs):
        return ne.utils.compose(inputs[0], inputs[1])

    def _single_affine_compose(self, inputs):
        affine_1 = affine_shift_to_identity(inputs[0])
        affine_2 = affine_shift_to_identity(inputs[1])
        composed = tf.linalg.matmul(affine_1, affine_2)
        return affine_identity_to_shift(composed)

    def compute_output_shape(self, input_shape):
        if self.return_affine:
            return (input_shape[0], self.ndims * (self.ndims + 1))
        else:
            return (input_shape[0], *self.volshape, self.ndims)


class AffineToDense(Layer):
    """
    Converts an affine transform to a dense shift transform. The affine must represent
    the shift between images (not over the identity).
    """

    def __init__(self, volshape, **kwargs):
        self.volshape = volshape
        self.ndims = len(volshape)
        super().__init__(**kwargs)

    def build(self, input_shape):

        shape = input_shape[1:]

        if len(shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if shape[0] != ex:
                raise ValueError('Expected flattened affine of len %d but got %d' % (ex, shape[0]))

        if len(shape) == 2 and (shape[0] != self.ndims or shape[1] != self.ndims + 1):
            raise ValueError('Expected affine matrix of shape %s but got %s' % (str((self.ndims, self.ndims + 1)), str(shape)))

        super().build(input_shape)

    def call(self, trf):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """

        return tf.map_fn(self._single_aff_to_shift, trf, dtype=tf.float32)

    def _single_aff_to_shift(self, trf):
        # go from vector to matrix
        if len(trf.shape) == 1:
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        trf += tf.eye(self.ndims + 1)[:self.ndims, :]  # add identity, hence affine is a shift from identity
        return ne.utils.affine_to_shift(trf, self.volshape, shift_center=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.volshape, self.ndims)


class InvertAffine(Layer):
    """
    Inverts an affine transform. The transform must represent
    the shift between images (not over the identity).
    """

    def build(self, input_shape):
        self.ndims = extract_affine_ndims(input_shape[1:])
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, trf):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """
        return tf.map_fn(self._single_invert, trf, dtype=tf.float32)

    def _single_invert(self, trf):
        matrix = affine_shift_to_identity(trf)
        inverse = tf.linalg.inv(matrix)
        return affine_identity_to_shift(inverse)


class AffineTransformationsToMatrix(Layer):
    """
    Computes the corresponding (flattened) affine from a vector of transform
    components. The components are in the order of (translation, rotation), so the
    input must a 1D array of length (ndim * 2).

    TODO: right now only supports 4x4 transforms - make this dimension-independent
    TODO: allow for scaling and shear components

    
    """

    def __init__(self, ndims, scale=False, shift=True, **kwargs):
        self.ndims = ndims
        self.scale = scale
        self.shift = shift
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('rigid registration is limited to 3D for now')

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, vector):
        """
        Parameters
            vector: tensor of affine components
        """
        return tf.map_fn(self._single_conversion, vector, dtype=tf.float32)

    def _single_conversion(self, vector):

        if self.ndims == 3:

            # extract components of input vector
            translation = vector[:3]
            angle_x = vector[3]/180*math.pi
            angle_y = vector[4]/180*math.pi
            angle_z = vector[5]/180*math.pi

            # x rotation matrix
            cosx  = tf.math.cos(angle_x)
            sinx  = tf.math.sin(angle_x)
            x_rot = tf.convert_to_tensor([
                [1,    0,     0],
                [0, cosx, -sinx],
                [0, sinx,  cosx]
            ], name='x_rot')

            # y rotation matrix
            cosy  = tf.math.cos(angle_y)
            siny  = tf.math.sin(angle_y)
            y_rot = tf.convert_to_tensor([
                [cosy,  0, siny],
                [0,     1,    0],
                [-siny, 0, cosy]
            ], name='y_rot')
            
            # z rotation matrix
            cosz  = tf.math.cos(angle_z)
            sinz  = tf.math.sin(angle_z)
            z_rot = tf.convert_to_tensor([
                [cosz, -sinz, 0],
                [sinz,  cosz, 0],
                [0,        0, 1]
            ], name='z_rot')

            # compose matrices
            t_rot = tf.tensordot(x_rot, y_rot, 1)
            m_rot = tf.tensordot(t_rot, z_rot, 1)

            # build scale matrix
            s = vector[6] if self.scale else 1.0
            m_scale = tf.convert_to_tensor([
                [s, 0, 0],
                [0, s, 0],
                [0, 0, s]
            ], name='scale')

        elif self.ndims == 2:
            # extract components of input vector
            translation = vector[:2]
            angle = vector[2]

            # rotation matrix
            cosz  = tf.math.cos(angle)
            sinz  = tf.math.sin(angle)
            m_rot = tf.convert_to_tensor([
                [cosz, -sinz],
                [sinz,  cosz]
            ], name='rot')

            s = vector[3] if self.scale else 1.0
            m_scale = tf.convert_to_tensor([
                [s, 0],
                [0, s]
            ], name='scale')

    
        # we want to encode shift transforms, so remove identity
        if self.shift:
            m_rot -= tf.eye(self.ndims)

        # scale the matrix
        m_rot = tf.tensordot(m_rot, m_scale, 1)

        # concat the linear translation
        matrix = tf.concat([m_rot, tf.expand_dims(translation, 1)], 1)

        # flatten
        affine = tf.reshape(matrix, [self.ndims * (self.ndims + 1)])
        
        return affine

# below are the core layers
class V2STransformer(Layer):
    """
    Volume to Slice Transformer (can simulate 2D multislice acquisition)

    @ author : allard wen shi, JHU SOM & ZJU, 2020/05

    """

    def __init__(self,
                 st_ratio,
                 interp_method='linear',
                 single_transform=False,
                 v2s=True,
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
        """
        self.st_ratio = st_ratio
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform
        self.v2s = v2s
        
        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('V2STransformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')
        
        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        self.vol_shape = input_shape[0][1:-1]
        self.trf_shape = input_shape[1][1:]
        self.nb_slice = input_shape[1][1]
        # confirm built
        self.built = True
        
    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # go from affine
        tf_vol = tf.map_fn(lambda x: self._v2s_transform(x, trf, vol.shape[1:-1]), vol, dtype=tf.float32)
                
        return tf_vol
        
    def _v2s_transform(self, vol, trf, volshape):
        
        trf = tf.reshape(trf, [self.nb_slice, self.ndims, self.ndims + 1])
        trf += tf.eye(self.ndims+1)[:self.ndims,:]  
        return nrn_utils.v2s_transform(vol, trf, volshape, 
                                       interp_method=self.interp_method, 
                                       st_ratio = self.st_ratio)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], *self.vol_shape[:2], self.trf_shape[0], 1)

class S2VTransformer(Layer):
    """
    Slice to Volume Transformer
    Scattered data interpolation (NN)
    Arguments: 
        st_ratio: slice thickness/recon isotropic resolution e.g., 4.0mm/0.8mm = 5
        nb_stack: num of stacks
        ort: orientation, only support 'axi','cor','sag', e.g., ['axi','cor','sag'] when nb_stack=3
        interp_method: 'linear' or 'nearest'
        v2s: if transform volume to slices
        sda_method: sda method

    @ author : allard wen shi, JHU SOM & ZJU, 2020/09
    """

    def __init__(self,
                 st_ratio,
                 nb_stack,
                 ort,
                 interp_method='nearest',
                 v2s=True,
                 sda_method=None,
                 **kwargs):
        """
        
            
        """
        self.st_ratio = st_ratio
        self.nb_stack = nb_stack
        self.ort = ort
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.v2s = v2s
        self.sda_method = sda_method
        
        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
       
        """

        if len(input_shape) > 2:
            raise Exception('S2VTransformer must be called on a list of length 2. Got {}'.format(len(input_shape)),
                            'First argument is the image, the second is the inverse rigid transform matrix.')
        
        # set up number of dimensions
        self.ndims = len(input_shape[0])-2
        self.inshape = input_shape
        self.imgshape = input_shape[0][1:-1]
        self.trf_shape = input_shape[1][1:]
        self.nb_slice = input_shape[0][-2] 
        self.z_scale = int(np.ceil(self.nb_slice*self.st_ratio/self.nb_stack))
        
        # as the nearest interpolation will have error at boundary
        self.vol_shape = (*self.imgshape[:-1], self.z_scale+2)
                
        self.built = True
        
    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol,trf = inputs[0], inputs[1]

        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])
        
        N_nda, D_nda = tf.map_fn(lambda x: self._s2v_transform(x, trf), vol, dtype=(tf.float32,tf.float32))   
        
        D_nda = tf.where(D_nda <= 0, tf.ones_like(D_nda), D_nda)
        
        return [N_nda, D_nda]
               
    def _s2v_transform(self, vol, trf):
        trf = tf.reshape(trf, [self.nb_slice, self.ndims, self.ndims + 1])
        trf += tf.eye(self.ndims+1)[:self.ndims,:]  
           
        return nrn_utils.s2v_transform(vol, trf, 
                                       nb_slice=self.nb_slice,
                                       volshape=self.vol_shape, 
                                       ort=self.ort,
                                       interp_method=self.interp_method, 
                                       st_ratio=self.st_ratio,
                                       nb_stack=self.nb_stack,
                                       sda_method=self.sda_method)
        
    def compute_output_shape(self, input_shape):        
        return [(input_shape[0][0], *self.vol_shape[:2],self.z_scale,1),(input_shape[0][0], *self.vol_shape[:2],self.z_scale,1)]
 

class BlurGaussian3D(Layer):
    """
    
    Blurring using Gaussian kernel convolution
    Arguments: 
        sigma: kernel standard deviation
        sda: choose if use scattered data approximation method

    """

    def __init__(self,
                 sigma=1,
                 sda=False,
                 **kwargs):
        self.sigma = sigma
        self.sda = sda
        self.ndims = None
        self.inshape = None
        
        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):

        if len(input_shape) > 2:
            raise Exception('Only perform at most two inputs!')
        
        # set up number of dimensions        
        self.inshape = input_shape
        self.ndims = len(input_shape[0]) - 2
        self.vol_shape = input_shape[0][1:-1]

        
        # confirm built
        self.built = True
        
    def call(self, inputs):

        # Gaussian filter
        blur_kernel = nrn_utils.gaussian_kernel([self.sigma]*self.ndims,windowsize=[7]*self.ndims)
        blur_kernel = tf.reshape(blur_kernel, blur_kernel.shape.as_list()+[1,1])
            
        if self.sda:
            assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
            
            N_nda, D_nda = inputs[0], inputs[1]
    
            # necessary for multi_gpu models...
            N_nda = K.reshape(N_nda, [-1, *self.inshape[0][1:]])
            D_nda = K.reshape(D_nda, [-1, *self.inshape[1][1:]])
                        
            N_nda = tf.nn.conv3d(N_nda, blur_kernel, [1, 1, 1, 1, 1], 'SAME')
            D_nda = tf.nn.conv3d(D_nda, blur_kernel, [1, 1, 1, 1, 1], 'SAME')
                   
            tf_vol = tf.math.divide(N_nda, D_nda+1e-8)
        
        else:
            tf_vol = inputs[0]
            tf_vol = tf.nn.conv3d(tf_vol, blur_kernel, [1, 1, 1, 1, 1], 'SAME')
        
        return tf_vol
                
    def compute_output_shape(self, input_shape):        
        return (input_shape[0][0], *self.vol_shape,1)
   
   
class OutlierReject(Layer):
    """
    This layer is aimed to not consider the outlier slices into the training session. 
    
    two inputs: (slices and idx to be discarded)

    """

    def __init__(self, **kwargs):
        """
        Parameters: 
        
            
        """
        self.inshape = None

        
        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):        
        self.inshape = input_shape
        self.nb_slice = input_shape[0][1]
        self.nb_idx = input_shape[1][0]
        
        # confirm built
        self.built = True
        
    def call(self, inputs):
        """
        Parameters
            inputs: 
        """
        params = inputs[0]
        idx = inputs[1]
        idx = tf.cast(idx,'int32')
        
        x = tf.gather(params,idx[0,:,0],axis=-2)
    
        return x
        
        
    def compute_output_shape(self, input_shape):        
        return (input_shape[0][0],input_shape[1][1],input_shape[0][2])

class SelfGeodesic(Layer):
    """    
    Convert to Geodesic distance metric
    
    # Arguments
        geo_type: choose geodesic type: SE3/SO3
        ndims: number of dimension
        norm_scale: a hyperparameter to balance the rotation and displacement metric

    """

    def __init__(self,
                 geo_type='SE3', 
                 ndims=3, 
                 norm_scale=24,
                 **kwargs):

        self.geo_type = geo_type
        self.ndims = ndims
        self.norm_scale = norm_scale
        
        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """


        """
        
        # set up number of dimensions        
        self.inshape = input_shape
        self.nb_slice = input_shape[1]
        
        assert self.nb_slice > 1,  ValueError('Number of Slice should be greater than 1!')
        
        # confirm built
        self.built = True
        
    def call(self, inputs):        
        geodesic_dist = tf.map_fn(lambda x: self._dist(x), inputs, dtype=tf.float32)   
             
        return geodesic_dist
                
    def _dist(self, transform):
                  
        delta_dist = []
        for i in range(int(self.nb_slice-1)):
            delta_dist += [self._geodesic(transform[i,:], transform[i+1,:])]
               
        delta_dist += [tf.zeros_like(delta_dist[0])]
        delta_dist[0] = tf.zeros_like(delta_dist[-1])
        delta_dist = tf.stack(delta_dist)
        delta_dist = tf.expand_dims(delta_dist, -1)
        
        return delta_dist
    
    def _geodesic(self, gt, pred):
        
        gt = tf.reshape(gt, [self.ndims, self.ndims + 1])
        pred = tf.reshape(pred,[self.ndims, self.ndims + 1])        
        rot_gt = gt[:3,:3]
        disp_gt = gt[:,3]
        rot_pred = pred[:3,:3]
        disp_pred = pred[:,3]                
        mat_dot = tf.matmul(K.transpose(rot_pred), rot_gt)
                
        term_rot = tf.clip_by_value(0.5*(tf.reduce_sum(tf.diag_part(mat_dot))-1.0),-1.0+1e-8,1.0-1e-8)
        term_rot = np.sqrt(2)*tf.abs(tf.acos(term_rot))
        term_rot = tf.cast(term_rot, tf.float32)      
        term_disp = tf.norm(disp_gt-disp_pred, ord=2)
        
        geodesic_dist = tf.sqrt(tf.square(term_rot)+tf.square(term_disp/self.norm_scale))

        return geodesic_dist
                
    def compute_output_shape(self, input_shape):        
        return (input_shape[0], self.nb_slice, 1)
    

class PReLU(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
        
        
    
    """

#     @interfaces.legacy_prelu_support
    def __init__(self, alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[2:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        pos = K.relu(inputs)
        if K.backend() == 'theano':
            neg = (K.pattern_broadcast(self.alpha, self.param_broadcast) *
                   (inputs - K.abs(inputs)) * 0.5)
        else:
            neg = -self.alpha * K.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
