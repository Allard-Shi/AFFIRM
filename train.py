#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used for training AFFIRM

\author   Wen Allard Shi, JHU SOM BME & ZJU BME 
\date     05/2020

"""

import os
import sys
import numpy as np
import pysitk.python_helper as ph
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from datetime import datetime, timedelta
from keras.callbacks import (ModelCheckpoint,LearningRateScheduler,
                            ReduceLROnPlateau, TensorBoard)
#import tensorflow.keras.backend as K
from keras.optimizers import Adam, RMSprop
import losses
from keras import backend as K
from keras.models import Model
from data_generators import data_generator
from networks import AFFIRM

sys.path.append('./ext/neuron')
sys.path.append('./ext/pynd-lib')
#from pynd import ndutils as nd
#import neuron.layers as nrn_layers
#import neuron.models as nrn_models

def lr_schedule(epoch, threshold=5):
    """Learning Rate Schedule

    Learning rate is rescheduled to be reduced after each epochs.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
        
    """
    decay = 0.05
    lr = 1e-4
    
    if epoch > threshold:
        lr = 1e-4/(1+decay*epoch)

    
    print('Learning rate: ', lr)
    return lr

################################# dir setting #################################
    
path = r'./compile/'
model_dir = r'./model/saved/'
config_dir = r'/model/config/'

################################# init params #################################

nb_slice = 24
ort = ['axi','cor','sag']
nb_stack = len(ort)
batch_size = 1
nb_epochs = 100
use_generator = True
abs_loc = True
load_inital_model = True
pixdim_recon = 0.8
s_thickness = 4.0
st_ratio = s_thickness/pixdim_recon
recurrent = 1
sigma,sigma_init = 0.52,0.8

###############################################################################

name_model = 'AFFIRM'                                                          # choose model
print('#'*40+' initial params '+'#'*40)
ph.print_info('\tNumber of Slice per Stack: {}'.format(nb_slice))
ph.print_info('\tNumber of Stack: {}'.format(nb_stack))
ph.print_info('\tStack Orientation: {}'.format(ort))
ph.print_info('\tSlice Thickness Ratio: %.2f' % st_ratio)
ph.print_info('\tNumber of Model Folding: {}'.format(recurrent))
ph.print_info('\tModel: {}'.format(name_model))
print('#'*96)
      
recon_train_data = np.load(path+'recon_train_data.npy')
recon_validation_data = np.load(path+'recon_validation_data.npy')

vol_shape = recon_train_data.shape[1:-1]
nb_subject = recon_train_data.shape[0]
print('number of training individual: %d'%nb_subject)

# generate motion and training data generator
train_generator = data_generator(recon_train_data, vol_shape, 
                                 nb_slice=nb_slice,st_ratio=st_ratio,
                                 batch_size=batch_size,
                                 nb_stack=nb_stack,
                                 model=name_model,
                                 ort=ort,
                                 abs_loc=abs_loc)

validation_generator = data_generator(recon_validation_data, 
                                      vol_shape, 
                                      nb_slice=nb_slice,
                                      st_ratio=st_ratio,
                                      nb_stack=nb_stack,
                                      model=name_model,
                                      ort=ort,
                                      abs_loc=abs_loc)


ph.print_info('\tVol Shape: {}'.format(vol_shape))
#steps_per_epoch = round(nb_subject/batch_size)

# gpu handling
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
config = tf.ConfigProto()
#off = rewriter_config_pb2.RewriterConfig.OFF
#config.graph_options.rewrite_options.memory_optimization = off
#config.gpu_options.allow_growth = True
#config.allow_soft_placement = True
K.set_session(tf.Session(config=config))

# init loss
loss_mse1 = losses.MSE().loss
loss_mse2 = losses.MSE().loss
loss_mse3 = losses.MSE().loss
loss_l21 = losses.Ln_norm().loss
loss_l22 = losses.Ln_norm().loss
loss_l23 = losses.Ln_norm().loss
loss_ncc = losses.NCC().loss

loss_continuity = losses.Continuity().loss
loss_grad = losses.Grad().loss
loss_geodesic = losses.Geodesic().loss

if name_model == 'AFFIRM':
        
    model = AFFIRM(vol_shape=vol_shape,
                   nb_slice=nb_slice,
                   st_ratio=st_ratio,
                   nb_stack=nb_stack,
                   ort=ort,
                   sigma=sigma,
                   sigma_init=sigma_init,
                   recurrent=recurrent)
    
#    losses = [loss_mse1, loss_mse2, loss_geodesic, loss_mse3]
    losses = [loss_mse1, loss_mse2]

#    weights = [0.0001,0.0005,1,1]
    weights = [1,3]

    lr = 6.41e-5
    
    print('sigma',sigma_init,sigma)
          
# Compile the model
model.compile(optimizer=Adam(lr=lr), 
              loss=losses,
              loss_weights=weights)

model.summary()
print('learning rate: ', lr)
print('loss weights: ', weights)

#lr_scheduler = LearningRateScheduler(lr_schedule)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='auto',
                              patience=3, min_lr=2e-7, verbose=1)

# prepare callbacks
save_file_name = os.path.join(model_dir, name_model+'_{epoch:02d}_last.h5')
save_callback = ModelCheckpoint(save_file_name)

if load_inital_model:
    load_model_file = r'./saved/xxxxxx.h5'
    model.load_weights(load_model_file, by_name=True)

    model.fit_generator(train_generator, 
                        epochs=nb_epochs,
                        initial_epoch=0,
                        callbacks=[save_callback, reduce_lr],
                        shuffle=True, 
                        steps_per_epoch=int(3600/(batch_size*nb_slice)),
#                        validation_data=([data_validation],[label_validation[...,3:],label_validation[...,3:]]),
                        validation_data=validation_generator,
                        validation_steps=int(1200/(batch_size*nb_slice)),
                        verbose=1)

