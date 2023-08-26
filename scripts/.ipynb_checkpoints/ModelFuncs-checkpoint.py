# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:23:22 2023

@author: JK-WORK
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS


def get_quipu_model(n_filters_block_1=64,kernel_size_block_1=7,dropout_intermediate_blocks=0.25,
                    n_filters_block_2=128,kernel_size_block_2=5,n_filters_block_3=256,kernel_size_block_3=3,
                    n_dense_1=1000,n_dense_2=500,dropout_final=0.4):
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
