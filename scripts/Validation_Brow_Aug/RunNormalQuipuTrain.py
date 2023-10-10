import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

import pandas as pd
import numpy as np
import tensorflow as tf
from ModelTrainer import ModelTrainer
import time
from ModelFuncs import get_quipu_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[2], 'GPU')

n_runs=20;
n1=2048;n2=1024;

'''
#Quipu reproduction
df_results = pd.DataFrame(0, index=np.arange(n_runs), columns=["Train Acc", "Validation acc", "Test Acc", "Runtime"])

for i in range(n_runs):
    start_time = time.time()
    mt=ModelTrainer()
    model=get_quipu_model(n_dense_1=n1,n_dense_2=n2);
    acc_train,acc_valid,acc_test=mt.quipu_def_train(model,n_epochs=60)
    runtime = time.time() - start_time 
    df_results.loc[i]=[acc_train,acc_valid,acc_test,runtime];

out_name='../results/QuipuReproduction_N1_'+ str(n1)+'_N2_'+str(n2)+'.csv';
df_results.to_csv(out_name, index=False)
print(df_results)

'''

df_results = pd.DataFrame(0, index=np.arange(n_runs), columns=["Train Acc", "Validation acc", "Test Acc", "Runtime"])

for i in range(n_runs):
    start_time = time.time()
    mt=ModelTrainer()
    model=get_quipu_model(n_dense_1=n1,n_dense_2=n2);
    acc_train,acc_valid,acc_test=mt.quipu_def_train(model,n_epochs=60,with_brow_aug=True)
    runtime = time.time() - start_time 
    df_results.loc[i]=[acc_train,acc_valid,acc_test,runtime];

out_name='../results/QuipuExtWBrowAug_N1_'+ str(n1)+'_N2_'+str(n2)+'.csv';
df_results.to_csv(out_name, index=False)
print(df_results)

