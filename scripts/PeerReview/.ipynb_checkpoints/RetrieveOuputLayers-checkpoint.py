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
best_model.summary()
X_train,X_valid,X_test,Y_train,Y_valid,Y_test=mt.dl.get_datasets_numpy_tuning_model(divide_for_tuning=False,tuning_valid_perc=0.15);

intermediate_model = tf.keras.Model(inputs=best_model.input, outputs=best_model.get_layer('activation').output)
softmax_model = tf.keras.Model(inputs=best_model.input, outputs=best_model.get_layer('softmax').output)
out_filters=intermediate_model.predict(X_test)
out_softmax=softmax_model.predict(X_test)

folder="../../results/PeerReview/RetrieveOutputs/"
np.save(folder+"out_softmax.npy",out_softmax)
np.save(folder+"out_filters.npy",out_filters)
np.save(folder+"X_test.npy",X_test)
np.save(folder+"y_true_np.npy",Y_test)
