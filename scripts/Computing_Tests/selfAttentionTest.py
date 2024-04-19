import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding,BatchNormalization, Softmax,Multiply, Attention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model


#Testing self attention block to see if it works

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')

input_trace = Input(shape=(700,1), dtype='float32', name='input')
first_conv = Conv1D(64, 7, padding="same",activation="relu")(input_trace)


x = Conv1D(64, 7, padding="same")(first_conv)
x = BatchNormalization(axis=1)(x)
x = Activation('relu')(x)
x = Conv1D(64, 7, padding="same")(x)

attention_probs = Softmax()(x)

attention_features=Multiply()([first_conv,attention_probs])

attention_features=Flatten()(attention_features)

output_barcode = Dense(8, activation='softmax', name='output_barcode')(attention_features)
model = Model(inputs=input_trace, outputs=output_barcode)


model.summary()
##This code shows that the computation of the probabilities is where we expect it! (last axis)
#In=np.shape(np.array([[[0, 2.0, 1.0],[1.0, 2.0, 1.0],[1.0, 2.0, 1.0],[3, 2, 1]]]))
#softmax_layer = tf.keras.layers.Softmax()
#softmax_layer(In)
