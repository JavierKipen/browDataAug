import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer;
from ModelFuncs import get_resnet_model,get_quipu_model
import operator
from datetime import datetime
import tensorflow as tf
from ResNetTuningUtils import crossval_run_w_notes
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("param")
# args = parser.parse_args()
# print(args.param);
# noise_std=args.param); #Parameter that we set from command line.

noise_std=0;
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')
mt=ModelTrainer(lr=1e-4,batch_size=256,opt_aug=False,use_weights=True,n_epochs_max=100,track_losses=True);
mt.da.stretch_std=0.08; 
mt.da.noise_std=noise_std; 
model=get_resnet_model(filter_size=64, block_layers=[3,2,2], init_conv_kernel=3,init_pool_size=3,dense_1=2048,dense_2=1024,activation_fnc='relu')
model.summary();
crossval_run_w_notes(mt,model,comment="Augmentation parameters check: Trying different noise std",title_file="Noise_std",n_runs=20)
