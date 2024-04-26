from DataLoader import DataLoader;
from DataAugmenterV2 import DataAugmenterV2;
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
import time
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import clone_model
import ipdb
import cupy as cp



class ModelTrainerV2():
    def __init__(self,n_epochs_max=50,lr = 1e-3,batch_size=128,brow_aug_use=True,track_losses=False, optimizer="Adam",momentum=None,validation_perc=0.1,decay_rate=1): 
        self.dl=DataLoader();
        self.da=DataAugmenterV2();
        self.shapeX = (-1, QUIPU_LEN_CUT,1); self.shapeY = (-1, QUIPU_N_LABELS);
        self.n_epochs_max=n_epochs_max;
        self.lr=lr;
        self.batch_size=batch_size;
        self.early_stopping_patience=n_epochs_max;
        self.brow_aug_use=brow_aug_use;
        self.train_losses=[];
        self.valid_losses=[];
        self.train_aug_losses=[];
        self.track_losses=track_losses;
        self.optimizer=optimizer;
        self.momentum=momentum;
        self.validation_perc=validation_perc;
        self.decay_rate=decay_rate;

    def train_es(self,model,tuning=True): #Runs training with early stopping. When tuning=true uses tuning test set, otherwise uses final test set.
        X_train,X_valid,X_test,Y_train,Y_valid,Y_test=self.dl.get_datasets_numpy_tuning_model(divide_for_tuning=tuning,tuning_valid_perc=self.validation_perc); #Gets the oversampled dataset (test set is same as quipu when not tuning, if not is separated from the not test data)
        if self.optimizer=="Adam":
            model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=self.lr),metrics = ['accuracy'])
        else:
            model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate=self.lr,momentum=self.momentum),metrics = ['accuracy'])
        X_valid_rs = X_valid.reshape(self.shapeX); Y_valid_rs = Y_valid.reshape(self.shapeY); #Reshapes for the model
        best_weights=model.get_weights();best_valid_loss=1e6;patience_count=0; #For patience setup
        self.train_losses=[];self.valid_losses=[];self.train_aug_losses=[]; #To keep track of the losses
        X_train_cupy=cp.asarray(X_train);#Converts to cupy for optimized augmentation
        #ipdb.set_trace();
        lr=self.lr;
        for n_epoch in range(self.n_epochs_max):
            print("=== Epoch:", n_epoch,"===")
            #Augmentation plus timing
            start_time = time.time()
            X_train_aug=self.da.augment(X_train_cupy); #Augments the data
            X=X_train_aug.get(); #Converts back to numpy
            del X_train_aug; #Erase cupy array, hope it works
            preparation_time = time.time() - start_time
            # Fit the model
            model.optimizer.lr.assign(lr) # Adjust learning rate!
            out_history = model.fit(x = X.reshape(self.shapeX), y = Y_train.reshape(self.shapeY), batch_size=self.batch_size, shuffle = True, epochs=1,verbose = 1 )
            #Computing valudation to check for early stopping
            print("Validation ds:")
            valid_res=model.evaluate(x = X_valid_rs,   y = Y_valid_rs,   verbose=True,batch_size=512);
            if valid_res[0]<best_valid_loss: ##In case of improvement, keeps best weights and resets patience counter
                best_valid_loss=valid_res[0]
                patience_count=0;
                best_weights=model.get_weights()
            else:
                patience_count+=1;
            training_time = time.time() - start_time - preparation_time
            #When keeping track of losses, computes them and saves them.
            if self.track_losses:
                train_res=model.evaluate(x = X_train.reshape(self.shapeX),   y = Y_train, batch_size=512);
                self.train_losses.append(train_res[0]);self.valid_losses.append(valid_res[0]);
                self.train_aug_losses.append(out_history.history['loss'][0]);
            
            print('  prep time: %3.1f sec' % preparation_time, '  train time: %3.1f sec' % training_time) ##Plots computation times
            if patience_count>self.early_stopping_patience or n_epoch==self.n_epochs_max-1: ##Stopping due to early stopping
                print("Stopping learning because of early stopping:")
                model.set_weights(best_weights)
                break
            lr=lr*self.decay_rate; #Decaying of learning rate
        train_acc,valid_acc,test_acc=self.eval_model_and_print_results(model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
        return train_acc, valid_acc, test_acc, n_epoch
    
    def train_w_all_data(self,model):
        ipdb.set_trace();
        X_train,X_test,Y_train,Y_test=self.dl.get_datasets_numpy_no_valid(); #Gets the oversampled dataset (test set is same as quipu when not tuning, if not is separated from the not test data)
        if self.optimizer=="Adam":
            model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=self.lr),metrics = ['accuracy'])
        else:
            model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate=self.lr,momentum=self.momentum),metrics = ['accuracy'])
        self.train_losses=[];self.train_aug_losses=[]; #To keep track of the losses
        X_train_cupy=cp.asarray(X_train);#Converts to cupy for optimized augmentation
        #ipdb.set_trace();
        lr=self.lr;
        for n_epoch in range(self.n_epochs_max):
            print("=== Epoch:", n_epoch,"===")
            #Augmentation plus timing
            start_time = time.time()
            X_train_aug=self.da.augment(X_train_cupy); #Augments the data
            X=X_train_aug.get(); #Converts back to numpy
            del X_train_aug; #Erase cupy array, hope it works
            preparation_time = time.time() - start_time
            # Fit the model
            model.optimizer.lr.assign(lr) # Adjust learning rate!
            out_history = model.fit(x = X.reshape(self.shapeX), y = Y_train.reshape(self.shapeY), batch_size=self.batch_size, shuffle = True, epochs=1,verbose = 1 )
            #Computing valudation to check for early stopping
            training_time = time.time() - start_time - preparation_time
            #When keeping track of losses, computes them and saves them.
            if self.track_losses:
                train_res=model.evaluate(x = X_train.reshape(self.shapeX),   y = Y_train, batch_size=512);
                self.train_losses.append(train_res[0]);
                self.train_aug_losses.append(out_history.history['loss'][0]);
            
            print('  prep time: %3.1f sec' % preparation_time, '  train time: %3.1f sec' % training_time) ##Plots computation times
            
            lr=lr*self.decay_rate; #Decaying of learning rate
            
        train_results= model.evaluate(x = X_train.reshape(self.shapeX), y = Y_train, verbose=False);
        test_results= model.evaluate(x = X_test.reshape(self.shapeX),  y = Y_test,  verbose=False)
        train_acc= train_results[1];test_acc= test_results[1];
        return train_acc, test_acc, n_epoch
    
    def reset_history(self):
        self.train_losses=[];self.valid_losses=[];self.train_aug_losses=[]; #To keep track of the losses    
    
    def eval_model_and_print_results(self,model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
        print("       [ loss , accuracy ]")
        train_results= model.evaluate(x = X_train.reshape(self.shapeX), y = Y_train, verbose=False);
        valid_results=model.evaluate(x = X_valid.reshape(self.shapeX),   y = Y_valid,   verbose=False)
        test_results= model.evaluate(x = X_test.reshape(self.shapeX),  y = Y_test,  verbose=False)
        
        print("Train:", train_results )
        print("Validation  :", valid_results )
        print("Test :", test_results )
        train_acc= train_results[1];valid_acc= valid_results[1];test_acc= test_results[1];
        return train_acc,valid_acc,test_acc
