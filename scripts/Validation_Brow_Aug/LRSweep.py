import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer
import tensorflow as tf
from ModelFuncs import get_quipu_model
import math
from datetime import datetime


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')


n_runs=2;
n1=2048;n2=1024;
brow_aug=0.9;
use_brow_aug=True;

lr=5e-4;batch_size=256; #Should keep them constant for all the runs to make a fair comparison


lr_str="{:.0E}".format(lr);
folder_es_train="../../results/CompareLRs/"+lr_str+"/";

if not os.path.exists(folder_es_train):
    os.makedirs(folder_es_train)# Create a new directory because it does not exist

mt=ModelTrainer(brow_std=brow_aug,batch_size=batch_size,brow_aug_use=use_brow_aug,lr=lr,opt_aug=False,n_epochs_max=1);
model=get_quipu_model(n_dense_1=n1,n_dense_2=n2);
str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
if use_brow_aug:
    run_name=str_time+"WBrowAug_"+str(int(brow_aug))+str(int(math.modf(brow_aug)[0]*100))+"_N1_"+str(n1)+"_N2_"+str(n2)+".csv";
else:
    run_name=str_time+"Reproduction"+"_N1_"+str(n1)+"_N2_"+str(n2)+".csv";
mt.crossval_es(model,n_runs=n_runs,data_folder=folder_es_train+run_name,save_each_row=True)
