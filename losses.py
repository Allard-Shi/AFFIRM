import sys
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        in_ch = J.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size  # TODO: simplify this
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return mean cc for each entry in batch
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return - self.ncc(y_true, y_pred)

class NCC2D:
    """
    Local (over window) normalized cross correlation loss.
    ignore the z axis information 
    
    
    
    
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * 2

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % int(ndims-1))
        in_ch = J.get_shape().as_list()[-1]
        nb_slice = J.get_shape().as_list()[-2]
        padding = 'SAME'
        cc = 0
        for i in range(nb_slice):
            
        # compute CC squares
            I2 = I[...,i,:] * I[...,i,:]
            J2 = J[...,i,:] * J[...,i,:]
            IJ = I[...,i,:] * J[...,i,:]
    
            # compute filters
            sum_filt = tf.ones([*self.win, in_ch, 1])
            strides = 1
            if ndims > 1:
                strides = [1] * (ndims + 1)
    
            # compute local sums via convolution
    
            I_sum = conv_fn(I[...,i,:], sum_filt, strides, padding)
            J_sum = conv_fn(J[...,i,:], sum_filt, strides, padding)
            I2_sum = conv_fn(I2, sum_filt, strides, padding)
            J2_sum = conv_fn(J2, sum_filt, strides, padding)
            IJ_sum = conv_fn(IJ, sum_filt, strides, padding)
    
            # compute cross correlation
            win_size = np.prod(self.win) * in_ch
            u_I = I_sum / win_size
            u_J = J_sum / win_size
    
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size  # TODO: simplify this
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    
            cc += cross * cross / (I_var * J_var + self.eps)
       
        cc = cc/nb_slice
        # return mean cc for each entry in batch
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return - self.ncc(y_true, y_pred)
    

class MSE:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def loss(self, y_true, y_pred):
        
        sr = K.square(y_true - y_pred)
        sr_nonzero = tf.cast(tf.count_nonzero(sr), tf.float32)       
        
        return 1.0 / (self.image_sigma**2) * K.sum(sr)/(sr_nonzero+1e-8)


class Ln_norm:
    """    
    Euclidean norm
    
    """

    def __init__(self, l=2):
        self.l = l

    def loss(self, y_true, y_pred):
        if self.l == 2:
            return  K.sqrt(K.sum(K.square(y_true - y_pred)))
        elif self.l == 1:
            return  K.sqrt(K.sum(K.abs(y_true - y_pred)))
        else:
            raise ('not provided, please wait for the next version')
    
class Geodesic_RO3:
    
    """
    Geodesic loss for rotation transform
    
    S0(3)
    SE(3)
    
    
    """
      
    def _to_matrix(self, vector):
                
       angle_x = vector[0]
       angle_y = vector[1]
       angle_z = vector[2]
      
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
    
       return m_rot

    def R(self,x):
    
        R1 = tf.eye(3) + tf.sin(x[2])* x[0] + (1.0 - tf.cos(x[2]))*K.dot(x[0], x[0])
        R2 = tf.eye(3) + tf.sin(x[3])* x[1] + (1.0 - tf.cos(x[3]))*K.dot(x[1], x[1])
    
        return K.dot(K.transpose(R1), R2)

    # Rodrigues' formula
    def get_theta(self,x):
    
        return tf.abs(tf.acos(tf.clip_by_value(0.5*(tf.reduce_sum(tf.diag_part(x))-1.0),-1.0+1e-7,1.0-1e-7)))
       

    def loss(self, y_true, y_pred):
        
        matrix1 = tf.map_fn(self._to_matrix,(y_true),dtype=tf.float32)
        matrix2 = tf.map_fn(self._to_matrix,(y_pred),dtype=tf.float32)
        
        angle_true = tf.sqrt(tf.reduce_sum(tf.square(matrix1), axis=1))
        angle_pred = tf.sqrt(tf.reduce_sum(tf.square(matrix2), axis=1))
        
        # compute axes
        axis_true = tf.nn.l2_normalize(y_true, dim=1)
        axis_pred = tf.nn.l2_normalize(y_pred, dim=1)
        
        # convert axes to corresponding skew-symmetric matrices
        proj = tf.constant(np.asarray([
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]]), dtype=tf.float32)
        
        skew_true = K.dot(axis_true, proj)
        skew_pred = K.dot(axis_pred, proj)
        skew_true = tf.map_fn(lambda x: tf.reshape(x, [3, 3]), skew_true)
        skew_pred = tf.map_fn(lambda x: tf.reshape(x, [3, 3]), skew_pred)

        # compute rotation matrices and do a dot product
        r = tf.map_fn(self.R,(skew_true,skew_pred,angle_true,angle_pred),dtype=tf.float32)
        
        # compute the angle error
        theta = tf.map_fn(self.get_theta, r)
        
        return tf.reduce_mean(theta)
    
class Geodesic:
    """
    measure the geodesic distance 
    
    @author: allard wen shi JHU SOM and ZJU BME
        
    """


    def __init__(self, input_type='vector', df=0, geo_type='SE3', ndims=3, norm_scale=48):
                
        # TODO: adapt to SO3
        self.df = df
        self.input_type = input_type
        self.geo_type = geo_type
        self.ndims = ndims
        self.norm_scale = norm_scale
                                    
    def _to_slice(self, inputs):
        """
        matrix shape: 12*1
        """
        
        y_true, y_pred = inputs[0], inputs[1]
#        if self.input_type != 'matrix':
#            matrix_true = tf.map_fn(self._to_matrix, y_true, dtype=tf.float32)
#            matrix_pred = tf.map_fn(self._to_matrix, y_pred, dtype=tf.float32)
            
        geodist = tf.map_fn(self._geodesic_dist, (y_true, y_pred), dtype=tf.float32)
        
        return geodist
        
    def _geodesic_dist(self, inputs):
        
        gt, pred = inputs[0], inputs[1]        
        gt = tf.reshape(gt, [3, 4])
        pred = tf.reshape(pred, [3, 4])
        rot_gt = gt[:3,:3]
        disp_gt = gt[:,3]
        rot_pred = pred[:3,:3]
        disp_pred = pred[:,3]
                
        mat_dot = tf.matmul(K.transpose(rot_pred), rot_gt)
        
        # the tf version we used did not have the automatic gradient calculation of tf.linalg.logm
#        mat_dot = tf.cast(mat_dot, tf.complex64)
#        term_rot = tf.norm(tf.real(tf.linalg.logm(mat_dot)), ord='fro', axis=(0,1)) 
        
        term_rot = tf.clip_by_value(0.5*(tf.reduce_sum(tf.diag_part(mat_dot))-1.0),-1.0+1e-8,1.0-1e-8)
        term_rot = 1.414*tf.abs(tf.acos(term_rot))
        term_rot = tf.cast(term_rot, tf.float32)      
        term_disp = tf.norm(disp_gt-disp_pred, ord=2)

        geodesic_dist = tf.sqrt(tf.square(term_rot)+tf.square(term_disp/self.norm_scale))

        return geodesic_dist
            
    def loss(self, y_true, y_pred):
        
        geodist = tf.map_fn(self._to_slice, (y_true, y_pred), dtype=tf.float32)
                
        return K.mean(geodist)
    
    
#        if self.df == 0:
#                       
#            return tf.reduce_mean(geodist)
#        
#        elif self.df == 2:
#            
#            return tf.reduce_mean(geodist)
#            
#        else:
#            raise ValueError('df is not valid!')

