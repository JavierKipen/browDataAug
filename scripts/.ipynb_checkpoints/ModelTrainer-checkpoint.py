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
from tensorflow.keras.optimizers import Adam


class ModelTrainer():
    def __init__(self):
        self.dl=DataLoader();
        self.da=DataAugmentator();
    
#    def quipu_improved_train(self):
        
    ##Quipu base code to compare
    def quipu_def_train(self,n_epochs=60):
        shapeX = (-1, QUIPU_LEN_CUT,1); shapeY = (-1, QUIPU_N_LABELS);
        #tensorboard, history = resetHistory()
        lr = 1e-3
        X_train,X_valid,Y_train,Y_valid,X_test,Y_test=self.dl.get_datasets_numpy_quipu();
        model=get_quipu_model ();
        model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = Adam(lr=0.001),
    metrics = ['accuracy']
)

        weights=class_weight.compute_class_weight(class_weight ='balanced',classes = np.arange(QUIPU_N_LABELS), y =np.argmax(Y_train,axis=1))
        weights=dict(zip(np.arange(QUIPU_N_LABELS), weights))
        for n in range(0, n_epochs):
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
            print('  loss: %5.3f' % out_history.history['loss'][0] ,'  acc: %5.4f' % out_history.history['accuracy'][0] ,'  val_acc: %5.4f' % out_history.history['val_accuracy'][0])
            #print('  loss: %5.3f' % out_history.history['loss'][0] ,'  acc: %5.4f' % out_history.history['accuracy'][0] ,'  val_acc: %5.4f' % out_history.history['val_acc'][0])
        print("       [ loss , accuracy ]")
        train_results= model.evaluate(x = X_train.reshape(shapeX), y = Y_train, verbose=False);
        valid_results=model.evaluate(x = X_valid.reshape(shapeX),   y = Y_valid,   verbose=False)
        test_results= model.evaluate(x = X_test.reshape(shapeX),  y = Y_test,  verbose=False)
        
        print("Train:", train_results )
        print("Validation  :", valid_results )
        print("Test :", test_results )
        train_acc= train_results[1];valid_acc= valid_results[1];test_acc= test_results[1];
        return train_acc, valid_acc, test_acc
            
    
        
    