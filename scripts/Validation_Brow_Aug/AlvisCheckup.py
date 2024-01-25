import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

import tensorflow as tf
from ModelFuncs import get_quipu_model
from ModelTrainer import ModelTrainer
import ipdb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
#tf.debugging.set_log_device_placement(True)


#mt=ModelTrainer(brow_aug_use=False,lr=2e-4,opt_aug=False,n_epochs_max=1);
#mt.train_es(model)



def activate_gpu():
    #x = tf.random.uniform((100, 1), minval=-1, maxval=1)
    #y = 2*x+1
    #inputs = tf.keras.Input(shape=(1,))
    #linear = tf.keras.layers.Dense(1)
    #outputs = linear(inputs)
    #model = tf.keras.Model(inputs=inputs, outputs=outputs, name="linear_model")
    #model.compile(
    #    loss=tf.keras.losses.MeanSquaredError(),
    #    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3),
    #)
    #history = model.fit(x, y, epochs=1, batch_size=x.shape[0])
    model=get_quipu_model(n_dense_1=512,n_dense_2=512);
    mt=ModelTrainer(brow_aug_use=False,lr=2e-4,opt_aug=False,n_epochs_max=1);
    mt.train_es(model)


#activate_gpu()

model=get_quipu_model(n_dense_1=512,n_dense_2=512);

mt=ModelTrainer(brow_aug_use=True,lr=2e-4,opt_aug=False,n_epochs_max=1);
#ipdb.set_trace()
mt.activate_gpu(model)
mt.train_es(model)

