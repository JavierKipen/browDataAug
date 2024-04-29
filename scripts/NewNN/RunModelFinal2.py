import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files


from ModelTrainerV2 import ModelTrainerV2;
from ModelFuncs import get_resnet_model,get_quipu_model,get_quipu_skipCon_model,get_AttResQuipu,ModelInfo
import operator
from datetime import datetime
import tensorflow as tf
from ModelTuningFuncs import crossval_run_w_notes
import argparse

##########################################################
#In this .py file we will run to the final test set!
parser = argparse.ArgumentParser()
parser.add_argument("param")
#args = parser.parse_args()
#print(args.param);
#param=float(args.param); #Parameter that we set from command line.

# +
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) #Memory growth crashes after many runs!


## Trying to control memory:
#Gb_limit=200; #This amount of Gb for tensorflow
#tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=Gb_limit*1024)])

#tf.config.gpu.set_per_process_memory_fraction(0.75) #https://github.com/tensorflow/tensorflow/issues/25138

#import cupy
#print(cupy.get_default_memory_pool().get_limit())  # 1073741824
# -

out_folder_newNN="../../results/NewNN/" #Result general folder
out_folder=out_folder_newNN+"QuipuRes/"; ##Corresponding subfolder

if not os.path.exists(out_folder):
    os.makedirs(out_folder)# Create a new directory because it does not exist

#Configs
comment="Training QuipuRes final with lower validation perc, Another LR. RELU. Adding attention"
tuning=False; #This makes it run on tuning df, and use the quipus test dataset
lr=2e-4;
batch_size=256;
n_epochs=100;
n_runs=500;

mt=ModelTrainerV2(lr=lr,batch_size=batch_size,track_losses=True,n_epochs_max=n_epochs,validation_perc=0.05);

# +
##Extra configs
#mt.da.stretch_rel_std=0.08;
#mt.da.brow_std=0.0001; #No brownian aug.

# +
#model,modelInfo=get_AttResQuipu(dropout_block=0.1,dense_2=512)

#model=get_quipu_model();
#modelInfo=ModelInfo(model_type="QuipuRes");

model,modelInfo=get_quipu_skipCon_model(add_attention=True,filter_size=64,kernels_blocks=[7,7,5,3,3],dropout_blocks=0.25,n_dense_1=2048,n_dense_2=1024,dropout_final=0.4,pool_size=3,activation="relu")


#model,modelInfo=get_quipu_skipCon_model();

# -

model.summary();


crossval_run_w_notes(mt,model,modelInfo,out_folder, title_file="QuipuResAttLowV",n_runs=n_runs,comment=comment,tuning=tuning,all_data=False)
