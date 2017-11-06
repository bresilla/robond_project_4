#!/usr/bin/env python3

import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools

def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides=strides)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    output_layer = bilinear_upsample(small_ip_layer)
    output_layer = layers.concatenate([output_layer, large_ip_layer])
    output_layer = separable_conv2d_batchnorm(output_layer, filters, strides=1)
    return output_layer

def fcn_model(inputs, num_classes):
    encode_layer_1 = encoder_block(inputs, 32, 2)
    encode_layer_2 = encoder_block(encode_layer_1, 64, 2)
    encode_layer_3 = encoder_block(encode_layer_2, 128, 2)
    convol_layer_1 = conv2d_batchnorm(encode_layer_3, 256, kernel_size=1, strides=1)
    decode_layer_1 = decoder_block(convol_layer_1, encode_layer_2, 128)
    decode_layer_2 = decoder_block(decode_layer_1, encode_layer_1, 64)
    decode_layer_3 = decoder_block(decode_layer_2, inputs, 32)
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decode_layer_3)

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

output_layer = fcn_model(inputs, num_classes)

learning_rate = 0.025
batch_size = 16
num_epochs = 50
steps_per_epoch = 250
validation_steps = 100
workers = 8

model = models.Model(inputs=inputs, outputs=output_layer)
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)

run_num = 'run_1'

val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')