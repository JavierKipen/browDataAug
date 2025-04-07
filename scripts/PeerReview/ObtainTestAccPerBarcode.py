# The models we saved were in the TF version of the container used to run them, so cannot load them. This script is to load them and save them as a .h5 file, so we can load them easily.

import numpy as np
import os
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("../../scripts/NewNN/")
sys.path.append("../../scripts/Common/")


from ModelTrainerV2 import ModelTrainerV2



model_path = "../../results/NewNN/Ablation/20240503_13-10-14_FinalLR2E4_BestModel.keras"

best_model = tf.keras.models.load_model(model_path)
mt=ModelTrainerV2();
X_train,X_valid,X_test,Y_train,Y_valid,Y_test=mt.dl.get_datasets_numpy_tuning_model(divide_for_tuning=False,tuning_valid_perc=0.15);

y_prob = best_model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)

y_classes_np=np.asarray(y_classes)
np.save("y_classes_np.npy",y_classes_np)
np.save("y_true_np.npy",Y_test)
