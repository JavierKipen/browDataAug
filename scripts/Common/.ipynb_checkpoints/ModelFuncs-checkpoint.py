# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:23:22 2023

@author: JK-WORK
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS


def get_quipu_model(n_filters_block_1=64,kernel_size_block_1=7,dropout_intermediate_blocks=0.25,
                    n_filters_block_2=128,kernel_size_block_2=5,n_filters_block_3=256,kernel_size_block_3=3,
                    n_dense_1=512,n_dense_2=512,dropout_final=0.4):
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')

    x = Conv1D(n_filters_block_1, kernel_size_block_1, padding="same")(input_trace)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_1, kernel_size_block_1, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x) 
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Conv1D(n_filters_block_2, kernel_size_block_2, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_2, kernel_size_block_2, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Conv1D(n_filters_block_3, kernel_size_block_3, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_3, kernel_size_block_3, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Flatten()(x)
    x = Dense(n_dense_1, activation='relu')(x)
    x = Dropout(dropout_final)(x)
    x = Dense(n_dense_2, activation='relu')(x)
    x = Dropout(dropout_final)(x)
    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model;
'''
Total params: 15,671,512
Trainable params: 15,667,472
Non-trainable params: 4,040  When Ndense1=2048 Ndense2=1024. To know how many params to start resnets with. 
'''


#based on https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
def get_resnet_model(filter_size=64, block_layers=[3,4,6,3], init_conv_kernel=7,init_pool_size=3, end_pool_size=2,dense_1=None,dropout_1=0.3,l2reg=None,activation_fnc='relu'):

    kernel_regularizer=None if (l2reg is None) else l2(l2reg);
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
    
    x = Conv1D(filter_size, init_conv_kernel, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(input_trace)
    x = BatchNormalization()(x)
    x = Activation(activation_fnc)(x)
    x = MaxPooling1D(pool_size=init_pool_size,strides=2, padding = 'same')(x)
    
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = resnet_identity_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc)
        else:
            # One Residual/Convolutional B
            filter_size = filter_size*2# The filter size will go on increasing by a factor of 2
            x = resnet_conv_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc)
            for j in range(block_layers[i] - 1):
                x = resnet_identity_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc)
    
    x = AveragePooling1D(pool_size=end_pool_size, padding = 'same')(x)
    x=Flatten()(x)
    if dense_1 is not None:
        x=Dense(dense_1, activation=activation_fnc,kernel_regularizer=kernel_regularizer)(x)
        x=Dropout(dropout_1)(x)

    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model;



def resnet_identity_block(x,filter_size,kernel_size=3,kernel_regularizer=None,activation_str='relu'):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',kernel_regularizer=kernel_regularizer)(x)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Activation(activation_str)(conv_block)
    # Layer 2
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',kernel_regularizer=kernel_regularizer)(conv_block)
    conv_block = BatchNormalization()(conv_block)
    # Add Residue
    out = layers.add([conv_block, x_skip])     
    out = Activation(activation_str)(out)
    return out

def resnet_conv_block(x,filter_size,kernel_size=3,kernel_regularizer=None,activation_str='relu'):
    # Layer 1
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(x)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Activation(activation_str)(conv_block)
    # Layer 2
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same')(conv_block)
    conv_block = BatchNormalization()(conv_block)
    # Add Residue
    x_skip = Conv1D(filter_size, 1, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(x) ##Kernel for skip connection is 1
    out = layers.add([conv_block, x_skip])     
    out = Activation(activation_str)(out)
    return x


if __name__ == "__main__":
    model=get_resnet_model(filter_size=64, block_layers=[3,4,6,3], init_conv_kernel=3,init_pool_size=3,l2reg=None,activation_fnc='relu')
    model.summary();

