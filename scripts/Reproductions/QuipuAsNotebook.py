import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer
import tensorflow as tf
from ModelFuncs import get_quipu_model
import pandas as pd
import math
from datetime import datetime
import ipdb
from tensorflow.keras.models import clone_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

#ipdb.set_trace()

subfolder_res= "QuipuAsNotebook/";
path_results="../../results/"+subfolder_res;

n_runs=10;
n_epochs=60;

if not os.path.exists(folder_es_train):
    os.makedirs(folder_es_train)# Create a new directory because it does not exist

mt=ModelTrainer();
model_base=get_quipu_model();

for i in range(n_runs):
    model=clone_model(model_base);
    
    train_acc, valid_acc, test_acc = mt.quipu_def_train(model,n_epochs=n_epochs,sameQuipuTestSet=True);
    str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
    run_name=str_time+"_QuipuAsNotebook_Ne_"+str(n_epochs) + ".csv";
    df_results = pd.DataFrame(0, index=[0], columns=["train_acc","valid_acc","test_acc"]);
    df_results["train_acc"]=train_acc; 
    df_results["valid_acc"]=valid_acc;
    df_results["test_acc"]=test_acc;
    df_results.to_csv(path_results+run_name, index=False)