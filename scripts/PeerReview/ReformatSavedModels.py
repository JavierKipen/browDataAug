#The models we saved were in the TF version of the container used to run them, so cannot load them. This script is to load them and save them as a .h5 file, so we can load them easily.

import numpy as np
import os
import pandas as pd
import tensorflow as tf


common_model_folder="../../results/NewNN/Ablation/";
#These are the models that were best of their batches when trained with all the parts.
models_filenames= ["20240503_11-42-14_FinalLR1E4_BestModel.keras","20240503_11-43-30_FinalLR1E4_BestModel.keras", 
                   "20240503_11-43-50_FinalLR1E4_BestModel.keras", "20240503_11-51-02_FinalLR2E4_BestModel.keras",
                   "20240503_12-05-49_FinalLR2E4_BestModel.keras", "20240503_13-10-14_FinalLR2E4_BestModel.keras",
                   "20240503_16-13-15_FinalLR5E4_BestModel.keras", "20240506_14-50-01_FinalLR2E4_BestModel.keras"]
models_paths=[common_model_folder+i for i in models_filenames];

for model_path in models_paths:
    rec_model = tf.keras.models.load_model(models_paths[0])
    rec_model.summary()
    rec_model.save(model_path[:-5]+"h5")
