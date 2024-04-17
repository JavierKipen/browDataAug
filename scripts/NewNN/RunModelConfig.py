import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files


from ModelTrainerV2 import ModelTrainerV2;
from ModelFuncs import get_resnet_model,get_quipu_model,get_quipu_skipCon_model
import operator
from datetime import datetime
import tensorflow as tf
from ModelTuningFuncs import crossval_run_w_notes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("param")
args = parser.parse_args()
print(args.param);
param=float(args.param); #Parameter that we set from command line.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')

#Configs
lr=1e-3;
batch_size=256;

mt=ModelTrainerV2(lr=lr,batch_size=batch_size,track_losses=True);

model,modelInfo=get_resnet_model(filter_size=128, block_layers=[4, 3, 3, 2], init_conv_kernel=3,init_pool_size=3,dense_1=1024,dropout_end=0.4,dropout_block= 0.1,activation_fnc='swish')

model.summary();
crossval_run_w_notes(mt,model,modelInfo,title_file="LR",n_runs=20)
