# The models we saved were in the TF version of the container used to run them, so cannot load them. This script is to load them and save them as a .h5 file, so we can load them easily.

import numpy as np
import os
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("../../scripts/NewNN/")
sys.path.append("../../scripts/Common/")


from ModelTrainerV2 import ModelTrainerV2


common_model_folder="../../results/NewNN/Ablation/";
#These are the models that were best of their batches when trained with all the parts.
models_filenames= ["20240503_11-42-14_FinalLR1E4_BestModel.keras","20240503_11-43-30_FinalLR1E4_BestModel.keras", 
                   "20240503_11-43-50_FinalLR1E4_BestModel.keras", "20240503_11-51-02_FinalLR2E4_BestModel.keras",
                   "20240503_12-05-49_FinalLR2E4_BestModel.keras", "20240503_13-10-14_FinalLR2E4_BestModel.keras",
                   "20240503_16-13-15_FinalLR5E4_BestModel.keras", "20240506_14-50-01_FinalLR2E4_BestModel.keras"]
models_paths=[common_model_folder+i for i in models_filenames];

for model_path in models_paths:
    print(model_path)
    rec_model = tf.keras.models.load_model(model_path)
    mt=ModelTrainerV2();
    X_train,X_valid,X_test,Y_train,Y_valid,Y_test=mt.dl.get_datasets_numpy_tuning_model(divide_for_tuning=False,tuning_valid_perc=0.15);
    mt.eval_model_and_print_results(rec_model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
