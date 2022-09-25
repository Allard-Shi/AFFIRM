'''
This utils file was copied from VoxelMorph

'''


import numpy as np
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import neuron as ne
import layers

def transform(img, trf, interp_method='linear', rescale=None):
    """
    Applies a transform to an image. Note that inputs and outputs are
    in tensor format i.e. (batch, *imshape, nchannels).
    """
    img_input = keras.Input(shape=img.shape[1:])
    trf_input = keras.Input(shape=trf.shape[1:])
    trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)
    y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, trf_scaled])
    return keras.Model([img_input, trf_input], y_img).predict([img, trf])

def is_affine(shape):
    return len(shape) == 1 or (len(shape) == 2 and shape[0] + 1 == shape[1])

def extract_affine_ndims(shape):
    if len(shape) == 1:
        # if vector, just compute ndims since length = N * (N + 1)
        return int((np.sqrt(4 * int(shape[0]) + 1) - 1) / 2)
    else:
        return int(shape[0])

def affine_shift_to_identity(trf):
    ndims = extract_affine_ndims(trf.shape.as_list())
    trf = tf.reshape(trf, [ndims, ndims + 1])
    trf = tf.concat([trf, tf.zeros((1, ndims + 1))], axis=0)
    trf += tf.eye(ndims + 1)
    return trf

def affine_identity_to_shift(trf):
    ndims = int(trf.shape.as_list()[-1]) - 1
    trf = trf - tf.eye(ndims + 1)
    trf = trf[:ndims, :]
    return tf.reshape(trf, [ndims * (ndims + 1)])

def gaussian_blur(tensor, level, ndims):
    """
    Blurs a tensor using a gaussian kernel (nothing done if level=1).
    """
    if level > 1:
        sigma = (level-1) ** 2
        blur_kernel = ne.utils.gaussian_kernel([sigma] * ndims)
        blur_kernel = tf.reshape(blur_kernel, blur_kernel.shape.as_list() + [1, 1])
        if ndims == 3:
            conv = lambda x: tf.nn.conv3d(x, blur_kernel, [1, 1, 1, 1, 1], 'SAME')
        else:
            conv = lambda x: tf.nn.conv2d(x, blur_kernel, [1, 1, 1, 1], 'SAME')
        return KL.Lambda(conv)(tensor)
    elif level == 1:
        return tensor
    else:
        raise ValueError('Gaussian blur level must not be less than 1')

