import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer;
from ModelFuncs import get_resnet_model,get_quipu_model,get_quipu_skipCon_model
import operator
from datetime import datetime
import tensorflow as tf
from ResNetTuningUtils import crossval_run_w_notes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("param")
args = parser.parse_args()
print(args.param);
dropout_final=float(args.param); #Parameter that we set from command line.

lr=4e-6
batch_size=256
noise_std=0.02;
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')
mt=ModelTrainer(lr=lr,batch_size=batch_size,use_weights=False,n_epochs_max=100,track_losses=True);

mt.da.stretch_std=0.05; 
mt.da.magnitude_std=0.04;
mt.da.stretch_prob=1; 
mt.da.brow_std=0.9; # we
mt.da.noise_std=noise_std; 
#model=get_resnet_model(filter_size=128, block_layers=[3,3,3,2], init_conv_kernel=3,init_pool_size=3,dense_1=None,dropout_val=0,activation_fnc='relu')
model,modelInfo=get_resnet_model(filter_size=128, block_layers=[4, 3, 3, 2], init_conv_kernel=3,init_pool_size=3,dense_1=1024,dropout_end=0.4,dropout_block= 0.1,activation_fnc='swish')
#model,modelInfo=get_quipu_skipCon_model(filter_size=64,kernels_blocks=[7,5,3],dropout_blocks=0.25,n_dense_1=2048,n_dense_2=1024,dropout_final=0.4,pool_size=3,activation="relu");
#model=get_quipu_model(n_dense_1=2048,n_dense_2=1024);

model.summary();
crossval_run_w_notes(mt,model,modelInfo,comment="Changing learning rate but with big data",title_file="LR_BD",n_runs=20)
