# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:04:26 2023

@author: JK-WORK
"""

from DataLoader import DataLoader;
from DataAugmentator import DataAugmentator;
from sklearn.utils import class_weight
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
from ModelFuncs import get_quipu_model
import time
import numpy as np

class ModelTrainer():
    def __init__(self):
        self.dl=DataLoader();
        self.da=DataAugmentator();
    
#    def quipu_improved_train(self):
        
    ##Quipu base code to compare
    def quipu_def_train(self):
        shapeX = (-1, QUIPU_LEN_CUT,1); shapeY = (-1, QUIPU_N_LABELS);
        #tensorboard, history = resetHistory()
        lr = 1e-3
        X_train,X_valid,Y_train,Y_valid,X_test,Y_test=self.dl.get_datasets_numpy_quipu();
        model=get_quipu_model ();
        weights=class_weight.compute_class_weight('balanced', np.arange(QUIPU_N_LABELS), Y_train)
        for n in range(0, 60):
            print("=== Epoch:", n,"===")
            start_time = time.time()
            X=self.da.quipu_augment(X_train);
            # Learning rate decay
            lr = lr*0.97
            model.optimizer.lr.assign(lr)
            preparation_time = time.time() - start_time
            # Fit the model
            out_history = model.fit( 
                x = X.reshape(shapeX), 
                y = Y_train.reshape(shapeY), 
                batch_size=32, shuffle = True,
                initial_epoch = n,  epochs=n+1,
                class_weight = weights, 
                validation_data=(X_valid.reshape(shapeX),  Y_valid.reshape(shapeY)),
                #callbacks = [tensorboard, history], 
                verbose = 0
            )
            training_time = time.time() - start_time - preparation_time
            
            # Feedback 
            print('  prep time: %3.1f sec' % preparation_time, 
                  '  train time: %3.1f sec' % training_time)
            print('  loss: %5.3f' % out_history.history['loss'][0] ,
                  '  acc: %5.4f' % out_history.history['acc'][0] ,
                  '  val_acc: %5.4f' % out_history.history['val_acc'][0])
        print("       [ loss , accuracy ]")
        print("Train:", model.evaluate(x = X_train.reshape(shapeX), y = Y_train, verbose=False) )
        print("Dev  :", model.evaluate(x = X_valid.reshape(shapeX),   y = Y_valid,   verbose=False) )
        print("Test :", model.evaluate(x = X_test.reshape(shapeX),  y = Y_test,  verbose=False) )
                
            
    
        
    