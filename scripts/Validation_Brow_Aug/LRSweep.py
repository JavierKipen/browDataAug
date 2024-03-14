import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer
import tensorflow as tf
from ModelFuncs import get_quipu_model
import math
from datetime import datetime
import ipdb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[2], 'GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

n_runs=200;
n1=2048;n2=1024;
brow_aug=0.9;
use_brow_aug=False;
red_train=False;
brow_stretch_check=False; ##To check if the stretching due to brow is the cause of the increase in acc!

lr=2e-3;batch_size=256; #Should keep them constant for all the runs to make a fair comparison

#ipdb.set_trace()
lr_str="{:.0E}".format(lr);

n_epochs=100; #100 epochs to check
use_weights=False; #We oversample using augmentation to achieve higher accuracies.
folder_res= "CompareLRsRed/" if red_train else "CompareLRs/";
folder_es_train="../../results/"+folder_res+lr_str+"/" ;

if red_train:
    n_epochs=50;
    use_weights=True; #weights should reduce considerably training time.

if not os.path.exists(folder_es_train):
    os.makedirs(folder_es_train)# Create a new directory because it does not exist

mt=ModelTrainer(brow_std=brow_aug,batch_size=batch_size,brow_aug_use=use_brow_aug,lr=lr,opt_aug=False,n_epochs_max=n_epochs,early_stopping_patience=n_epochs+1,use_weights=use_weights,check_stretch_brow=brow_stretch_check);
model=get_quipu_model(n_dense_1=n1,n_dense_2=n2);
str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
if brow_stretch_check:
    run_name=str_time+"StretchCheck"+"_N1_"+str(n1)+"_N2_"+str(n2)+".csv";
elif use_brow_aug:
    run_name=str_time+"WBrowAug_"+str(int(brow_aug))+str(int(math.modf(brow_aug)[0]*100))+"_N1_"+str(n1)+"_N2_"+str(n2)+".csv";
else:
    run_name=str_time+"Reproduction"+"_N1_"+str(n1)+"_N2_"+str(n2)+".csv";

#mt.activate_gpu(model)
mt.crossval_es(model,n_runs=n_runs,data_folder=folder_es_train+run_name,save_each_row=True)
