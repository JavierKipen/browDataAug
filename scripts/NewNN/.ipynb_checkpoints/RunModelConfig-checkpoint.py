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
#args = parser.parse_args()
#print(args.param);
#param=float(args.param); #Parameter that we set from command line.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

out_folder_newNN="../../results/NewNN/" #Result general folder
out_folder=out_folder_newNN+"Test/"; ##Corresponding subfolder

if not os.path.exists(out_folder):
    os.makedirs(out_folder)# Create a new directory because it does not exist

#Configs
lr=1e-3;
batch_size=256;
n_epochs=50;
n_runs=100;

mt=ModelTrainerV2(lr=lr,batch_size=batch_size,track_losses=True,n_epochs_max=n_epochs);

model,modelInfo=get_resnet_model(filter_size=32, block_layers=[2,1,1], init_conv_kernel=3,init_pool_size=3,dense_1=10,dropout_end=0.4,dropout_block= 0.1,activation_fnc='relu')

model.summary();


crossval_run_w_notes(mt,model,modelInfo,out_folder, title_file="LR",n_runs=n_runs)
