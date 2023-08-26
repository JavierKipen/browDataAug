import pandas as pd
import numpy as np
import tensorflow as tf
from ModelTrainer import ModelTrainer
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

n_runs=20;

df_results = pd.DataFrame(0, index=np.arange(n_runs), columns=["Train Acc", "Validation acc", "Test Acc", "Runtime"])

for i in range(n_runs):
    start_time = time.time()
    mt=ModelTrainer()
    acc_train,acc_valid,acc_test=mt.quipu_def_train(n_epochs=60)
    runtime = time.time() - start_time 
    df_results.loc[i]=[acc_train,acc_valid,acc_test,runtime];

df_results.to_csv('../results/QuipuReproduction.csv', index=False)
print(df_results)
