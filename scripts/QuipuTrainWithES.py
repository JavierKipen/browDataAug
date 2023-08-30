from ModelTrainer import ModelTrainer
import tensorflow as tf
from ModelFuncs import get_quipu_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
mt=ModelTrainer();
model=get_quipu_model();
mt.crossval_es(model,n_runs=20)