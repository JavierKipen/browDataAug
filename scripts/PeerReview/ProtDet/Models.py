import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding,BatchNormalization, Softmax,Dot, Attention,Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
from tensorflow.keras.regularizers import l2
import ipdb



def YupanaNetModif(filter_size=64,kernels_blocks=[7,5,3],dropout_blocks=0.25,n_dense_1=512,n_dense_2=512,dropout_final=0.4,pool_size=3,activation="relu",add_attention=False):
    #modelInfo=ModelInfo(model_type="QuipuSkip",filter_size=filter_size,kernels_blocks=kernels_blocks,dense_1=n_dense_1,dense_2=n_dense_2,dropout_end=dropout_final,dropout_blocks=dropout_blocks,activation=activation);
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
    x=input_trace;
    if add_attention:
        x = Conv1D(filter_size, 7, padding = 'same')(x);
        x = BatchNormalization(axis=1)(x)
        x = Activation(activation)(x)
        x = self_attention_block(x,filter_size,activation_str=activation)
    
    for i in range(len(kernels_blocks)):
        x = quipu_block_skip_con(x,filter_size,kernels_blocks[i],pool_size,dropout_blocks,activation)
        filter_size*=2;
    x = Flatten()(x)
    x = Dense(n_dense_1, activation=activation)(x)
    x = Dropout(dropout_final)(x)
    x = Dense(n_dense_2, activation=activation)(x)
    x = Dropout(dropout_final)(x)
    output_barcode = Dense(1, activation='softmax', name='output_antibody')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model;