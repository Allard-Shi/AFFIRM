"""

The relevant networks 

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020
"""

import sys
import math
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
from keras.models import Model
from keras.layers import (Conv3D, Activation, Input, UpSampling3D, concatenate, SimpleRNN,
                          BatchNormalization, LSTM, ELU, Permute, ConvLSTM2D, Reshape, Add,
                          GRU, Bidirectional)
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
from keras.regularizers import l2

prepath = r'./'

# import neuron layers
sys.path.append(prepath+'/ext/neuron')
sys.path.append(prepath+'/ext/pynd-lib')
sys.path.append(prepath+'/ext/pytools-lib')
sys.path.append(prepath+'/network')


import neuron.layers as nrn_layers
import neuron.models as nrn_models
import neuron.utils as nrn_utils

# other vm functions
import layers

from ext.neuron import neuron as ne
from utils import gaussian_blur

 
def piabstanh(x):
    rot = 135 * K.tanh(x)
    return rot   

def pirelatanh(x):
    rot = 10 * K.tanh(x)
    return rot   

def mc_rot_tanh(x):    
    rot =  70 * K.tanh(x)
    return rot 

def mc_disp_tanh(x):    
    rot =  15* K.tanh(x)
    return rot 


def mt_sigmoid(x, alpha=2.):    
    y = alpha * K.sigmoid(x)

    return y

def pitanh(x):    
    """
    activation in Singh, Ayush et al., TMI, 2020
        
    """
    rot = 180 * K.tanh(x)
    return rot   
    
def acptanh(x):
    """
    activation modified in Hou et al., TMI, 2018
        
    """
    acp = 90 * K.tanh(x)
    return acp 

     
def MC_Model(vol_shape,
             nb_slice,
             st_ratio,
             recurrent_units=2304,
             recurrent=False,
             nb_units=1024,
             nb_units_fc=256,
             src_feats=1,
             name='mc_model'):
    """
    motion correction block in the model  
    
    """ 
   
    # init params
    ndims = len(vol_shape)     
    Conv = getattr(KL, 'Conv%dD' % int(ndims-1))
        
    # input slice         
    source_input = Input(shape=(*vol_shape[:2], nb_slice, src_feats))

    # permute slices loc (z)
    permute_source = Permute((3,1,2,4),name=name+'_permute_input')(source_input)
    
    x = TimeDistributed(Conv(32, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer="he_normal",name=name+'_conv1'))(permute_source)
    x = BatchNormalization(axis=-1,name=name+'_bn1')(x)
    x = Activation('relu',name=name+'_relu1')(x)
    x = TimeDistributed(Conv(32, (3, 3), kernel_initializer="he_normal",padding='same',name=name+'_conv2'))(x)
    x = BatchNormalization(axis=-1,name=name+'_bn2')(x)
    x = Activation('relu',name=name+'_relu2')(x)
           
    for i, filter_size in enumerate([64,64,128,128,128]):
        x = TimeDistributed(Conv(filter_size, (3, 3), padding='same', 
                                 kernel_initializer="he_normal",name=name+'_conv%d'%(i+3)))(x)
        x = BatchNormalization(axis=-1,name=name+'_bn%d'%(i+3))(x)
        x = Activation('relu', name=name+'_relu%d'%(i+3))(x)
        
        if i < 4:
            x = TimeDistributed(KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'_maxpooling%d'%(i+3)))(x)
        else:
            x = TimeDistributed(KL.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpooling'))(x)
            
    x = TimeDistributed(KL.Flatten(name=name+'_flatten'))(x)
    
    # bidirectional network
        
    bdiri_rnn = Bidirectional(GRU(nb_units, return_sequences=True, unroll=True, name=name+"_bidir_rnn"))   
#    bdiri_rnn_disp = Bidirectional(GRU(512, return_sequences=True, unroll=True, name=name+"_bidir_rnn_disp"))   

    encoder_outputs = bdiri_rnn(x)
#    encoder_outputs_disp = bdiri_rnn_disp(x)
    
    # rotation fully connected layers
    encoder_outputs = TimeDistributed(KL.Dropout(0.4),
                                      name=name+"_fc_dropout")(encoder_outputs)
    
#    encoder_outputs_disp = TimeDistributed(KL.Dropout(0.4),
#                                      name=name+"_fc_dropout_disp")(encoder_outputs_disp)
    
    rot_x = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_x_fc")(encoder_outputs)    
    rot_y = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_y_fc")(encoder_outputs)
    rot_z = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_z_fc")(encoder_outputs)
        
    rot_x = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_x")(rot_x)   
    rot_y = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_y")(rot_y)  
    rot_z = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_z")(rot_z)
    
    disp_x = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_x_fc")(encoder_outputs)
    disp_y = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_y_fc")(encoder_outputs)
    disp_z = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_z_fc")(encoder_outputs)
                        
    disp_x = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_x")(disp_x)   
    disp_y = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_y")(disp_y)  
    disp_z = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_z")(disp_z)

    rotation = KL.concatenate([rot_x, rot_y],name=name+"_rotation_temp")
    rotation = KL.concatenate([rotation, rot_z],name=name+"_rotation")
                
    displacement = KL.concatenate([disp_x, disp_y],name=name+"_displacement_temp")
    displacement = KL.concatenate([displacement, disp_z],name=name+"_displacement")

    transform = KL.concatenate([displacement, rotation],name=name+"_pred_transform")

    return Model(inputs=[source_input], outputs=[transform], name=name)
            
    
def MC_Model_pro(vol_shape,
                 nb_slice,
                 st_ratio,
                 nb_channel_2d=[64,64,128,128,128],
                 nb_channel_3d=[64,64,128,128,128],
                 recurrent=False,
                 nb_units=1024,
                 nb_units_fc=256,
                 dropout_rate=0.4,
                 src_feats=1,
                 nb_size=(12,12,8),
                 fusion_type=True,
                 name='mc_model'):
    """
    motion correction block in the model  
    
    """ 
   
    # init params
    ndims = len(vol_shape)     
    Conv = getattr(KL, 'Conv%dD' % int(ndims-1))
        
    # input slice         
    source_input = Input(shape=(*vol_shape[:2], nb_slice, src_feats))

#    x0 = Input(shape=(*nb_size,nb_channel_3d[1]),name=name+'_x0')
#    x1 = Input(shape=(*nb_size,nb_channel_3d[2]),name=name+'_x1')

    x2 = Input(shape=(*nb_size,nb_channel_3d[3]),name=name+'_x2')
#    x3 = Input(shape=(6,6,4,nb_channel_3d[3]),name=name+'_x3')
#    x4 = Input(shape=(3,3,2,nb_channel_3d[4]),name=name+'_x4')

    volume = [x2]

    # permute slices loc (z)
    permute_source = Permute((3,1,2,4),name=name+'_permute_input')(source_input)
    
    x = TimeDistributed(Conv(32, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer="he_normal",name=name+'_conv1'))(permute_source)
    x = BatchNormalization(axis=-1,name=name+'_bn1')(x)
    x = Activation('relu',name=name+'_relu1')(x)
    x = TimeDistributed(Conv(32, (3, 3), kernel_initializer="he_normal",padding='same',name=name+'_conv2'))(x)
    x = BatchNormalization(axis=-1,name=name+'_bn2')(x)
    x = Activation('relu',name=name+'_relu2')(x)
          
    for i, filter_size in enumerate(nb_channel_2d):
        x = TimeDistributed(Conv(filter_size, (3, 3), padding='same', 
                                 kernel_initializer="he_normal",
                                 name=name+'_conv%d'%(i+3)))(x)
        x = BatchNormalization(axis=-1,name=name+'_bn%d'%(i+3))(x)
        x = Activation('relu', name=name+'_relu%d'%(i+3))(x)
        
       
        if fusion_type != None and i == 3:
            inplane_size = volume[0].shape.as_list()[1:3]
                
            x_temp = feat_fusion(nb_slice, 
                                 inplane_size=inplane_size, 
                                 nb_channel=filter_size, 
                                 nb_size=nb_size,
                                 fusion_type=fusion_type,
                                 name=name+'_fusion%d'%i)([x,volume[0]])
            
            print('test shape:',x.shape,x_temp.shape)
                    
            
#            x = KL.Add(name=name+'_add')([x,x_temp])            
#            x = KL.Concatenate(name=name+'_concat')([x,x_temp])            
            
            
        if i == 4:
            x = TimeDistributed(KL.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpooling'))(x)         
        else:           
            x = TimeDistributed(KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'_maxpooling%d'%(i+3)))(x)

            
#                x_temp = TimeDistributed(KL.Flatten(name=name+'_flatten_temp'))(x_temp)
#                x = KL.Concatenate(name=name+'_fusionconcat_%d'%i)([x,x_temp])
    
#            x = TimeDistributed(KL.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpooling'))(x)

    x = TimeDistributed(KL.Flatten(name=name+'_flatten'))(x)
    x_temp = TimeDistributed(KL.Flatten(name=name+'_flatten_temp'))(x_temp)
#    x = KL.Concatenate(name=name+'_mixture',axis=-1)([x, x_temp])

#    # bidirectional network
    bdiri_rnn = Bidirectional(GRU(nb_units, return_sequences=True, unroll=True, name=name+"_bidir_rnn"))   
#    bdiri_rnn_disp = Bidirectional(GRU(512, return_sequences=True, unroll=True, name=name+"_bidir_rnn_disp"))   

    if recurrent:        
        encoder_outputs = bdiri_rnn(x)
#        encoder_outputs_disp = bdiri_rnn_disp(x)
        
#        print('encoder_outputs shape',encoder_outputs.shape)  
        if fusion_type != None:
            inputs = [source_input]+volume
        else:
            inputs = [source_input]
#        x5 = KL.Flatten(name=name+'_flatten_info3d')(x4)
#        info_3d = KL.Lambda(lambda x: tf.stack([x for k in range(nb_slice)],axis=-2), 
#                            name=name+'_expand')(x4)
#        info_3d = x_temp
            
        encoder_outputs = KL.Concatenate(name=name+'_mixture',axis=-1)([x_temp, encoder_outputs])
        print('dropout_rate:', dropout_rate)
        encoder_outputs = TimeDistributed(KL.Dropout(dropout_rate),name=name+"_fc_dropout")(encoder_outputs)   
#        encoder_outputs_disp = TimeDistributed(KL.Dropout(0.4),name=name+"_fc_dropout_disp")(encoder_outputs_disp)
        
#        encoder_outputs = KL.Concatenate(name=name+'_mixture',axis=-1)([encoder_outputs, info_3d])
#        encoder_outputs_disp = KL.Concatenate(name=name+'_mixture_disp',axis=-1)([encoder_outputs_disp, info_3d])
#    
                
        print('encoder_outputs shape',encoder_outputs.shape)      
#
        
    else:
        encoder_outputs = bdiri_rnn(x)
        inputs = [source_input]


    # rotation fully connected layers        
    rot_x = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_x_fc")(encoder_outputs)    
    rot_y = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_y_fc")(encoder_outputs)
    rot_z = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_z_fc")(encoder_outputs)
        
    rot_x = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_x")(rot_x)   
    rot_y = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_y")(rot_y)  
    rot_z = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_z")(rot_z)
    
    disp_x = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_x_fc")(encoder_outputs)
    disp_y = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_y_fc")(encoder_outputs)
    disp_z = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_z_fc")(encoder_outputs)
                        
    disp_x = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_x")(disp_x)   
    disp_y = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_y")(disp_y)  
    disp_z = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_z")(disp_z)

    rotation = KL.concatenate([rot_x, rot_y],name=name+"_rotation_temp")
    rotation = KL.concatenate([rotation, rot_z],name=name+"_rotation")
                
    displacement = KL.concatenate([disp_x, disp_y],name=name+"_displacement_temp")
    displacement = KL.concatenate([displacement, disp_z],name=name+"_displacement")

    transform = KL.concatenate([displacement, rotation],name=name+"_pred_transform")

    return Model(inputs=inputs, outputs=[transform], name=name)


def MC_Model_lastconcat(vol_shape,
                         nb_slice,
                         st_ratio,
                         nb_channel_2d=[64,64,128,128,128],
                         nb_channel_3d=[64,64,128,128,128],
                         recurrent=False,
                         nb_units=1024,
                         nb_units_fc=256,
                         dropout_rate=0.4,
                         src_feats=1,
                         nb_size=(12,12,8),
                         fusion_type=True,
                         name='mc_model'):
    """
    motion correction block in the model  
    
    """ 
   
    # init params
    ndims = len(vol_shape)     
    Conv = getattr(KL, 'Conv%dD' % int(ndims-1))
        
    # input slice         
    source_input = Input(shape=(*vol_shape[:2], nb_slice, src_feats))

#    x0 = Input(shape=(*nb_size,nb_channel_3d[1]),name=name+'_x0')
#    x1 = Input(shape=(*nb_size,nb_channel_3d[2]),name=name+'_x1')

    x2 = Input(shape=(*nb_size,nb_channel_3d[3]),name=name+'_x2')
#    x3 = Input(shape=(6,6,4,nb_channel_3d[3]),name=name+'_x3')
#    x4 = Input(shape=(3,3,2,nb_channel_3d[4]),name=name+'_x4')

    volume = [x2]

    # permute slices loc (z)
    permute_source = Permute((3,1,2,4),name=name+'_permute_input')(source_input)
    
    x = TimeDistributed(Conv(32, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer="he_normal",name=name+'_conv1'))(permute_source)
    x = BatchNormalization(axis=-1,name=name+'_bn1')(x)
    x = Activation('relu',name=name+'_relu1')(x)
    x = TimeDistributed(Conv(32, (3, 3), kernel_initializer="he_normal",padding='same',name=name+'_conv2'))(x)
    x = BatchNormalization(axis=-1,name=name+'_bn2')(x)
    x = Activation('relu',name=name+'_relu2')(x)
          
    for i, filter_size in enumerate(nb_channel_2d):
        x = TimeDistributed(Conv(filter_size, (3, 3), padding='same', 
                                 kernel_initializer="he_normal",
                                 name=name+'_conv%d'%(i+3)))(x)
        x = BatchNormalization(axis=-1,name=name+'_bn%d'%(i+3))(x)
        x = Activation('relu', name=name+'_relu%d'%(i+3))(x)
    
            
            
        if i == 4:
            x = TimeDistributed(KL.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpooling'))(x)         
        else:           
            x = TimeDistributed(KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=name+'_maxpooling%d'%(i+3)))(x)
            
#                x_temp = TimeDistributed(KL.Flatten(name=name+'_flatten_temp'))(x_temp)
#                x = KL.Concatenate(name=name+'_fusionconcat_%d'%i)([x,x_temp])
    
#            x = TimeDistributed(KL.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpooling'))(x)

    x = TimeDistributed(KL.Flatten(name=name+'_flatten'))(x)
    x2 = KL.AveragePooling3D((4,4,4), strides=(4,4,4), name=name+'_avgpooling_3d')(x2)  
    x2 = KL.Flatten(name=name+'_flatten_temp')(x2)
#    x = KL.Concatenate(name=name+'_mixture',axis=-1)([x, x_temp])

#    # bidirectional network
    bdiri_rnn = Bidirectional(GRU(nb_units, return_sequences=True, unroll=True, name=name+"_bidir_rnn"))   
#    bdiri_rnn_disp = Bidirectional(GRU(512, return_sequences=True, unroll=True, name=name+"_bidir_rnn_disp"))   

    if recurrent:        
        encoder_outputs = bdiri_rnn(x)
#        encoder_outputs_disp = bdiri_rnn_disp(x)
        
#        print('encoder_outputs shape',encoder_outputs.shape)  
        if fusion_type != None:
            inputs = [source_input]+volume
        else:
            inputs = [source_input]
#        x5 = KL.Flatten(name=name+'_flatten_info3d')(x4)
        info_3d = KL.Lambda(lambda x: tf.stack([x for k in range(nb_slice)],axis=-2), 
                            name=name+'_expand')(x2)
       
            
        encoder_outputs = KL.Concatenate(name=name+'_mixture',axis=-1)([info_3d, encoder_outputs])
        print('dropout_rate:', dropout_rate)
        encoder_outputs = TimeDistributed(KL.Dropout(dropout_rate),name=name+"_fc_dropout")(encoder_outputs)   
#        encoder_outputs_disp = TimeDistributed(KL.Dropout(0.4),name=name+"_fc_dropout_disp")(encoder_outputs_disp)
        
#        encoder_outputs = KL.Concatenate(name=name+'_mixture',axis=-1)([encoder_outputs, info_3d])
#        encoder_outputs_disp = KL.Concatenate(name=name+'_mixture_disp',axis=-1)([encoder_outputs_disp, info_3d])
#    
                
        print('encoder_outputs shape',encoder_outputs.shape)      
#
        
    else:
        encoder_outputs = bdiri_rnn(x)
        inputs = [source_input]


    # rotation fully connected layers        
    rot_x = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_x_fc")(encoder_outputs)    
    rot_y = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_y_fc")(encoder_outputs)
    rot_z = TimeDistributed(KL.Dense(nb_units_fc, activation='relu'),
                            name=name+"_estimation_rotation_z_fc")(encoder_outputs)
        
    rot_x = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_x")(rot_x)   
    rot_y = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_y")(rot_y)  
    rot_z = TimeDistributed(KL.Dense(1, activation=mc_rot_tanh),
                            name=name+"_estimation_rotation_z")(rot_z)
    
    disp_x = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_x_fc")(encoder_outputs)
    disp_y = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_y_fc")(encoder_outputs)
    disp_z = TimeDistributed(KL.Dense(128, activation='relu'),
                            name=name+"_estimation_displacement_z_fc")(encoder_outputs)
                        
    disp_x = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_x")(disp_x)   
    disp_y = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_y")(disp_y)  
    disp_z = TimeDistributed(KL.Dense(1, activation=mc_disp_tanh),
                            name=name+"_estimation_displacement_z")(disp_z)

    rotation = KL.concatenate([rot_x, rot_y],name=name+"_rotation_temp")
    rotation = KL.concatenate([rotation, rot_z],name=name+"_rotation")
                
    displacement = KL.concatenate([disp_x, disp_y],name=name+"_displacement_temp")
    displacement = KL.concatenate([displacement, disp_z],name=name+"_displacement")

    transform = KL.concatenate([displacement, rotation],name=name+"_pred_transform")

    return Model(inputs=inputs, outputs=[transform], name=name)

def recon(vol_shape, nb_slice, st_ratio, nb_stack, 
          ort, name, nb_channel=[64,64,64,128], src_feats=1):
    
    ndims = len(vol_shape)     
    Conv = getattr(KL, 'Conv%dD' % int(ndims))
    
    vol_input = Input(shape=(*vol_shape[:2], 120, src_feats), name="vol_input")

    x = Conv(32, 7, strides=2, padding='same', kernel_initializer="he_normal", name=name+'_conv1')(vol_input)
    x = BatchNormalization(axis=-1,name=name+'_bn1')(x)
    x = Activation('relu',name=name+'_relu1')(x)
    x = Conv(32, 3, kernel_initializer="he_normal", padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(axis=-1,name=name+'_bn2')(x)
    x = Activation('relu',name=name+'_relu2')(x)

    for i, channel_size in enumerate(nb_channel):
        x = Conv(channel_size, 3, strides=1, padding='same', name=name+'_conv%d'%(i+3))(x)
        x = KL.BatchNormalization(axis=-1)(x)
        x = KL.Activation('relu',name=name+'_relu%d'%(i+3))(x)
      
        if i < 3:
            x = KL.MaxPooling3D(2, strides=2, padding='same', name=name+'_maxpooling%d'%(i+3))(x)
        else:
            x = KL.AveragePooling3D((4,4,4), strides=(4,4,4), name=name+'_avgpooling')(x)
             
    output_feat = KL.Flatten(name=name+'_flatten')(x)
    
    print('output_feat shape:', output_feat.shape)
    
    return Model(inputs=[vol_input], outputs=[output_feat], name=name)
    
def recon_pro(vol_shape, nb_slice, st_ratio, nb_stack, 
          ort, name='recon_pro', nb_channel=[64,64,128,128,128], src_feats=1):
    
    ndims = len(vol_shape)     
    Conv = getattr(KL, 'Conv%dD' % int(ndims))
    MP = getattr(KL, 'MaxPooling%dD' % int(ndims))
    
    vol_input = Input(shape=(*vol_shape[:2], 120, src_feats), name="vol_input")
    # 192*192*120
    x = Conv(32, 7, strides=2, padding='same', kernel_initializer="he_normal", name=name+'_conv1')(vol_input)
    x = BatchNormalization(axis=-1,name=name+'_bn1')(x)
    x = Activation('relu',name=name+'_relu1')(x)
    x = Conv(32, 3, kernel_initializer="he_normal", padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(axis=-1,name=name+'_bn2')(x)
    x = Activation('relu',name=name+'_relu2')(x)

    # 96*96*60
    x = Conv(nb_channel[0], 3, strides=1, padding='same', name=name+'_conv3')(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu',name=name+'_relu3')(x)
    x = MP(2, strides=2, padding='same', name=name+'_maxpooling3')(x)
    
    # 48*48*30
    x = Conv(nb_channel[1], 3, strides=1, padding='same', name=name+'_conv4')(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu',name=name+'_relu4')(x)
    x1 = MP(2, strides=2, padding='same', name=name+'_maxpooling4')(x)
       
    # 24*24*15
    x = Conv(nb_channel[2], 3, strides=1, padding='same', name=name+'_conv5')(x1)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu',name=name+'_relu5')(x)
    x2 = MP(2, strides=2, padding='same', name=name+'_maxpooling5')(x)

    # 12*12*8
    x = Conv(nb_channel[3], 3, strides=1, padding='same', name=name+'_conv6')(x2)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu',name=name+'_relu6')(x)
    
    return Model(inputs=[vol_input], outputs=[x], name=name)


def channel_multiply(x):
    """
    channel-wise multiplication
    """
    
    x1, x2 = x    
    x1_shape, x2_shape = x1.shape, x2.shape
    x1_shape, x2_shape = x1_shape.as_list(), x2_shape.as_list()    
    nb_extend = int(len(x1_shape)-len(x2_shape))
        
    # TODO not beautiful 
    for i in range(nb_extend):
        x2 = tf.expand_dims(x2,-2)
            
    return x1*x2 

def cosine_dist(x):
    """
    matrix multiplication
    
    """
    x1, x2 = x
    
    return tf.matmul(x1,x2)

def broadcast_concatenate(x):
  
    x1, x2 = x     
    x1_shape = x1.shape
    x1_shape = x1_shape.as_list()   
    new_array = [tf.concat((x1[...,i,:],x2), axis=-1) for i in range(x1_shape[1])]

    return tf.stack(new_array, axis=1)
        
def feat_fusion(nb_slice,
                inplane_size,
                nb_channel,
                nb_size,
                name,
                fusion_type='nonlocal',
                nb_head=8,
                nb_reduction=1,
                regularization=0.001):  
    """    
    feature fusion flow
    nonlocal fusion denotes as the affinity fusion method
      
    mmtm:
        
    Joze, Hamid Reza Vaezi, et al. "MMTM: Multimodal Transfer Module for CNN Fusion." 
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
            
    """
    assert fusion_type in ['mmtm','nlnn','multihead','nonlocal'], \
    ValueError('The fusion type {} is not provided. '.format(fusion_type))
    
    input_slice =  Input(shape=(nb_slice, *nb_size[:2], nb_channel))    
    input_volume = Input(shape=(*nb_size, nb_channel))
        
    if fusion_type == 'nlnn':
        
        # 36*128
        key = KL.Conv3D(int(nb_channel/nb_reduction), 1, padding='same', use_bias=False, name=name+'_key_conv')(input_volume)
        key = KL.Reshape((-1, int(nb_channel/nb_reduction)), name=name+'_key_reshape')(key)
        key = KL.Permute((2,1), name=name+'_key_permute')(key)

        print('key shape:',key.shape)

        # 36*128
        value = KL.Conv3D(int(nb_channel/nb_reduction), 1, padding='same', use_bias=False, name=name+'_value_conv')(input_volume)
        value = KL.Reshape((-1, int(nb_channel/nb_reduction)), name=name+'_value_reshape')(value)
        
        # 24*9*128
        query = TimeDistributed(KL.Conv2D(int(nb_channel/nb_reduction), 1, padding='same', use_bias=False, name=name+'_query_conv'))(input_slice)
        query = TimeDistributed(KL.Reshape((-1, int(nb_channel/nb_reduction)), name=name+'_query_reshape'))(query)
            
        print('query shape:',query.shape)

        # 24*9*36
        output_slice = KL.Lambda(lambda x: tf.matmul(*x)/math.sqrt(int(nb_channel/nb_reduction)))([query, key])
        
        # TODO scale
        output_slice = TimeDistributed(KL.Softmax(axis=-1, name=name+'_softmax'))(output_slice)
        print('output_slice shape:',output_slice.shape)

        output_slice = KL.Lambda(lambda x: tf.matmul(*x))([output_slice, value])
        print('output_slice shape:',output_slice.shape)
    
        output_slice = TimeDistributed(KL.Reshape((*inplane_size,nb_channel), name=name+'_slice_reshape'))(output_slice)
#        output_slice = TimeDistributed(KL.Conv2D(nb_channel, 1, padding='same', use_bias=False, name=name+'_slice_conv'))(output_slice)

        print('output_slice shape:',output_slice.shape)

    if fusion_type == 'nonlocal':
        
        # 36*128
        key = KL.Conv3D(nb_channel, 1, padding='same', use_bias=False, name=name+'_key_conv')(input_volume)
        key = KL.Reshape((-1, nb_channel), name=name+'_key_reshape')(key)
        key = KL.Permute((2,1), name=name+'_key_permute')(key)

        print('key shape:',key.shape)

        # 36*128
        value = KL.Conv3D(nb_head, 1, padding='same', use_bias=False, name=name+'_value_conv')(input_volume)
        value = KL.Reshape((-1, nb_head), name=name+'_value_reshape')(value)
        
        # 24*9*128
        query = TimeDistributed(KL.Conv2D(nb_channel, 1, padding='same', use_bias=False, name=name+'_query_conv'))(input_slice)
        query = TimeDistributed(KL.Reshape((-1, nb_channel), name=name+'_query_reshape'))(query)
            
        print('query shape:',query.shape)

        # 24*9*36
#        output_slice = KL.Lambda(lambda x: tf.matmul(*x)/math.sqrt(nb_channel))([query, key])
                
        output_slice = KL.Lambda(lambda x: tf.linalg.matmul(*x)/math.sqrt(nb_channel))([query, key])

        # TODO scale
        output_slice = TimeDistributed(KL.Softmax(axis=-1, name=name+'_softmax'))(output_slice)
        
        print(output_slice.shape, value.shape)
        
        output_slice = KL.Lambda(lambda x: tf.linalg.matmul(*x))([output_slice, value])
#        output_slice = TimeDistributed(KL.Reshape((*nb_size[:2], nb_channel), name=name+'_affinity_reshape'))(output_slice)
#        output_slice = TimeDistributed(KL.Conv2D(8, 1, padding='same', use_bias=False, name=name+'_principle'))(output_slice)

        print('output_slice shape new:',output_slice.shape)
         
    if fusion_type == 'multihead':

        multihead = []        
        for i in range(nb_head):
        
            # 36*128
            key = KL.Conv3D(nb_channel, 1, padding='same', use_bias=False, name=name+'_key_conv%d'%i)(input_volume)
            key = KL.Reshape((-1, nb_channel), name=name+'_key_reshape%d'%i)(key)
            key = KL.Permute((2,1), name=name+'_key_permute%d'%i)(key)
        
            # 36*128
            value = KL.Conv3D(nb_channel, 1, padding='same', use_bias=False, name=name+'_value_conv%d'%i)(input_volume)
            value = KL.Reshape((-1, nb_channel), name=name+'_value_reshape%d'%i)(value)
            
            # 24*9*128
            query = TimeDistributed(KL.Conv2D(nb_channel, 1, padding='same', use_bias=False, name=name+'_query_conv%d'%i))(input_slice)
            query = TimeDistributed(KL.Reshape((-1, nb_channel), name=name+'_query_reshape%d'%i))(query)
                    
            # 24*9*36
            output_slice = KL.Lambda(lambda x: tf.matmul(*x))([query, key])
            
            # TODO scale
            output_slice = Activation('softmax')(output_slice)
            print('output_slice shape:',output_slice.shape)
       
            multihead += [KL.Lambda(lambda x: tf.matmul(*x))([output_slice, value])]
                
       
        multihead = KL.Lambda(lambda x: tf.concat(x,axis=-1))(multihead)
        print('multihead shape:',multihead.shape)
        
        multihead = TimeDistributed(KL.Conv1D(nb_channel, 1, padding='same', use_bias=False, name=name+'_multihead_conv%d'%i))(multihead)        
        output_slice = TimeDistributed(KL.Reshape((*inplane_size,nb_channel), name=name+'_multihead_reshape%d'%i))(multihead)

        print('multihead shape:',output_slice.shape)
        
    if fusion_type == 'mmtm':
        
        x1 = TimeDistributed(KL.GlobalAveragePooling2D(name=name+'_slice_avgpooling'))(input_slice)
        x2 = KL.GlobalAveragePooling3D(name=name+'_volume_avgpooling')(input_volume)
        
        # fusion
        Z = KL.Lambda(broadcast_concatenate)([x1, x2])
        Z = TimeDistributed(KL.Dense(nb_channel, 
                                     name=name+'_fusion'))(Z)
        
        x1 = TimeDistributed(KL.Dense(nb_channel,
                                      name=name+'_slice_split'))(Z)    
    
        x1 = Activation(mt_sigmoid)(x1)
            
        output_slice = KL.Lambda(channel_multiply)([input_slice, x1])
        
    return Model(inputs=[input_slice, input_volume], outputs=[output_slice], name=name+'_fusion_flow')

           
def AFFIRM(vol_shape,
           nb_slice,
           st_ratio,
           nb_stack,
           ort,
           nb_units=1024,
           src_feats=1,
           sigma=0.5,
           sigma_init=1,
           recurrent=1,
           fusion_type='nonlocal',
           predict=False,
           name='fetalMCSRRNet_RNN'):
    
    
    """    
    AFFIRM network
    
    
    
    
    
    """ 
        
    ndims = len(vol_shape) 
        
    # inputs check
    assert sigma < 2 and sigma > 0, 'sigma should set above 0 and below 3.'
    assert ndims == 3,  'Only 3D situation is under consideration.'
    
    source_input = Input(shape=(*vol_shape[:2],int(nb_slice*nb_stack), src_feats), name=name+'_input_stacks')
    stacks = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=nb_stack, axis=-2), name=name+'_split_stack')(source_input)
    
    volume = Input(shape=(*vol_shape[:2],int(nb_slice*st_ratio), src_feats), name=name+'_volume')
    
    transformation_input = Input(shape=(int(nb_slice*nb_stack), 6), name=name+'_input_transformation')
#    displacement = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1),name='extract_disp')(transformation_input)
#    displacement_xyz = displacement[0]
       
    idx_validslice = Input(shape=(None,1),name=name+'_valid_slice_idx')
                
     #  Set transformation    
    sda = sda_flow(vol_shape=vol_shape, nb_slice=nb_slice, st_ratio=st_ratio, 
                   nb_stack=nb_stack, ort=ort, name='sda_flow')
    
    rnn_model = recon_pro(vol_shape=vol_shape, 
                          nb_slice=nb_slice, 
                          st_ratio=st_ratio, 
                          nb_stack=nb_stack, 
                          ort=ort, name='3D_branch')
    
    print(rnn_model.summary())
    
    # reconstruction volume
#    recon_vol = sda([source_input,transformation_input])
#    recon_vol = layers.BlurGaussian3D(sigma=sigma_init,sda=True, name=name+'_gaussian_blur_0')(recon_vol)
    
    recon_vol = volume
    mc_model = []
    for i in range(nb_stack):
        mc_model += [MC_Model_pro(vol_shape=vol_shape, 
                                     nb_slice=nb_slice, 
                                     st_ratio=st_ratio, 
                                     nb_units=nb_units, 
                                     src_feats=src_feats,
                                     recurrent=True, 
                                     fusion_type=fusion_type,
                                     name=name+'_mc_model_'+str(i))]
        
    print(mc_model[0].summary())
        
    for i in range(recurrent):
    
#        print('info3d', info_3d.shape)
#        print('real info 3d:',len(info_3d))
#
#        info_3d = KL.Lambda(lambda x: tf.stack([x for k in range(nb_slice)],axis=-2), 
#                            name=name+'_expand_%d' % i)(info_3d)
#            
#        print('real info 3d:',info_3d.shape)
 
        for j in range(nb_stack):  
            info_3d = rnn_model([recon_vol])

            if j == 0:
                if nb_stack == 1:            
                    mc_block = mc_model[j]
                    transformation = mc_block([stacks,info_3d])
#                    transformation = mc_block([stacks]+info_3d)
                else:
                    mc_block = mc_model[j]
                    print('stacks', stacks[j].shape)

                    transformation = mc_block([stacks[j],info_3d])

#                    transformation = mc_block([stacks[j]]+info_3d)
                
            else:          
                mc_block = mc_model[j]
                transformation = KL.concatenate([transformation, mc_block([stacks[j],info_3d])],axis=-2)
#                transformation = KL.concatenate([transformation, mc_block([stacks[j]]+info_3d)],axis=-2)
                       
#        transformation = KL.concatenate([displacement_xyz,rotation],axis=-1)
                
                
        # fast and iterative approximate the fetal brain volume        
        recon_vol = sda([source_input,transformation])
                
        # set sigma in reconstruction (descending scheme)
        sigma_temp = sigma_init-(i+1)*(sigma_init-sigma)/recurrent 
        
        # blur the reconsturction result
        recon_vol = layers.BlurGaussian3D(sigma=sigma_temp,
                                          sda=True,
                                          name=name+'_gaussian_blur_%d'%(i+1))(recon_vol) 
        
        
    displacement, rotation = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1),name=name+'_split_transform')(transformation)

    # TODO: to adpat outlier rejection into the reconstruction    
    if not predict:
        
         # uncomment the code and comment out the current part when using geodesic distance loss
#        transformation = layers.OutlierReject()([transformation,idx_validslice])   
#        transformation = TimeDistributed(layers.AffineTransformationsToMatrix(ndims, shift=False, name=name+'_matrix_conversion_last'))(transformation)

#        stack_transform = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=nb_stack, axis=-2),name=name+'_split_stack_transform')(transformation)
#        
#        geodesic_se3 = []
#        
#        for i in range(nb_stack):
#            geodesic_se3 += [layers.SelfGeodesic(name=name+'_geo_%d'%i)(stack_transform[i])]
#            
#        geodesic = KL.Lambda(lambda x: tf.concat(x, axis=1), name=name+'_geo_stack')(geodesic_se3)
                        
        rotation = layers.OutlierReject()([rotation,idx_validslice])  
        displacement = layers.OutlierReject()([displacement,idx_validslice])  
#        geodesic = layers.OutlierReject()([geodesic,idx_validslice])  
        
        return Model(inputs=[source_input, transformation_input, idx_validslice, volume], 
                     outputs=[rotation, displacement])    
                
    else:
        return Model(inputs=[source_input, volume], 
                     outputs=[transformation])  
        
def sda_flow(vol_shape,
             nb_slice,
             st_ratio,
             nb_stack,
             ort,
             name,
             src_feats=1):
    """
    scatter data interpolation flow
    
    """
    ndims = len(vol_shape) 
    source_input = Input(shape=(*vol_shape[:2],int(nb_slice*nb_stack), src_feats), name=name+"_input_stacks")
    transformation_input = Input(shape=(int(nb_slice*nb_stack), 6),name=name+'_input_transformation')
    
    
    matrix = TimeDistributed(layers.AffineTransformationsToMatrix(ndims, name=name+'_matrix_conversion'))(transformation_input)
    matrix_inv = TimeDistributed(layers.InvertAffine(name=name+'_inverse_matrix'))(matrix)
        
    N_sda, D_nda = layers.S2VTransformer(st_ratio=st_ratio, 
                                       nb_stack=nb_stack, 
                                       ort=ort,name=name+'_s2v_transformer')([source_input, matrix_inv])
        
    return Model(inputs=[source_input, transformation_input], outputs=[N_sda, D_nda], name=name)

# def fetalMCSRRNet(vol_shape,
#                   nb_slice,
#                   st_ratio,
#                   nb_stack,
#                   ort,
#                   nb_units=1024,
#                   src_feats=1,
#                   sigma=0.52,
#                   sigma_init=0.8,
#                   recurrent=1,
#                   predict=False,
#                   name='fetalMCSRRNet'):
    
    
#     """
    
#     Recurrent model-based registration slice-to-volume registration
    
    
#     """ 
        
#     ndims = len(vol_shape) 
        
#     # inputs check
#     assert sigma < 2 and sigma > 0, 'sigma should set above 0 and below 3.'
#     assert ndims == 3,  'Only 3D situation is under consideration.'
    
#     source_input = Input(shape=(*vol_shape[:2],int(nb_slice*nb_stack), src_feats), name=name+'_input_stacks')
#     stacks = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=nb_stack, axis=-2), name=name+'_split_stack')(source_input)
           
#     idx_validslice = Input(shape=(None,1),name=name+'_valid_slice_idx')
                
#      #  Set transformation    
#     sda = sda_flow(vol_shape=vol_shape, nb_slice=nb_slice, st_ratio=st_ratio, 
#                    nb_stack=nb_stack, ort=ort, name='sda_flow')
    
    
#     # reconstruction volume
# #    recon_vol = sda([source_input,transformation_input])
# #    recon_vol = layers.BlurGaussian3D(sigma=sigma_init,sda=True, name=name+'_gaussian_blur_0')(recon_vol)
             
#     mc_model = []
#     for i in range(nb_stack):
#         mc_model += [MC_Model(vol_shape=vol_shape, nb_slice=nb_slice, st_ratio=st_ratio, 
#                               nb_units=nb_units, src_feats=src_feats,recurrent=True, 
#                               name=name+'_mc_model_'+str(i))]
        
#     print(mc_model[0].summary())
        
#     for i in range(recurrent):
    
# #        print('real info 3d:',len(info_3d))
# #
# #        info_3d = KL.Lambda(lambda x: tf.stack([x for k in range(nb_slice)],axis=-2), 
# #                            name=name+'_expand_%d' % i)(info_3d)
# #            
# #        print('real info 3d:',info_3d.shape)
 
#         for j in range(nb_stack):        
#             if j == 0:
#                 if nb_stack == 1:            
#                     mc_block = mc_model[j]
#                     transformation = mc_block([stacks])
# #                    transformation = mc_block([stacks]+info_3d)
#                 else:
#                     mc_block = mc_model[j]
#                     print('stacks', stacks[j].shape)

#                     transformation = mc_block([stacks[j]])

# #                    transformation = mc_block([stacks[j]]+info_3d)
                
#             else:          
#                 mc_block = mc_model[j]
#                 transformation = KL.concatenate([transformation, mc_block([stacks[j]])],axis=-2)
# #                transformation = KL.concatenate([transformation, mc_block([stacks[j]]+info_3d)],axis=-2)
                       
# #        transformation = KL.concatenate([displacement_xyz,rotation],axis=-1)
                
#         # fast and iterative approximate the fetal brain volume        
#         recon_vol = sda([source_input,transformation])
                
#         # set sigma in reconstruction (descending scheme)
#         sigma_temp = sigma_init-(i+1)*(sigma_init-sigma)/recurrent 
        
#         # blur the reconsturction result
#         recon_vol = layers.BlurGaussian3D(sigma=sigma_temp,
#                                           sda=True,
#                                           name=name+'_gaussian_blur_%d'%(i+1))(recon_vol)    
        
#     displacement, rotation = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1),name=name+'_split_transform')(transformation)

#     # TODO: to adpat outlier rejection into the reconstruction    
#     if not predict:
        
# #        transformation = layers.OutlierReject()([transformation,idx_validslice])   
        
# #        transformation = TimeDistributed(layers.AffineTransformationsToMatrix(ndims, shift=False, name=name+'_matrix_conversion_last'))(transformation)


# #        stack_transform = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=nb_stack, axis=-2),name=name+'_split_stack_transform')(transformation)
# #        
# #        geodesic_se3 = []
# #        
# #        for i in range(nb_stack):
# #            geodesic_se3 += [layers.SelfGeodesic(name=name+'_geo_%d'%i)(stack_transform[i])]
# #            
# #        geodesic = KL.Lambda(lambda x: tf.concat(x, axis=1), name=name+'_geo_stack')(geodesic_se3)
                        
#         rotation = layers.OutlierReject()([rotation,idx_validslice])  
#         displacement = layers.OutlierReject()([displacement,idx_validslice])  
# #        geodesic = layers.OutlierReject()([geodesic,idx_validslice])  
        
#         return Model(inputs=[source_input, idx_validslice], 
#                      outputs=[rotation, displacement])    
        

        
#     else:
#         return Model(inputs=[source_input], 
#                      outputs=[transformation])  
