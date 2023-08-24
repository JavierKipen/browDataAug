import pandas as pd
import numpy as np
import os
from params import QUIPU_DATA_FOLDER,QUIPU_VALIDATION_PROP_DEF
from DatasetFuncs import allDataset_loader,dataset_split

class DataLoader():
    def __init__(self,min_perc_test=4,max_perc_test=15,reduce_dataset_samples=None):
        self.min_perc_test=min_perc_test;
        self.max_perc_test=max_perc_test; #Porcentages for the split between test and train ds.
        self.reduce_dataset_samples=reduce_dataset_samples;
        self.df_cut=allDataset_loader(QUIPU_DATA_FOLDER,cut=True); #Loads both datasets to the class
        self.df_uncut=allDataset_loader(QUIPU_DATA_FOLDER,cut=False);
        
    def get_datasets_numpy_quipu(self,validation_prop=QUIPU_VALIDATION_PROP_DEF): #Gets the numpy arrays for the NN as it is done in Quipus code, with train, validation and test sets
        df_train,df_test=dataset_split(self.df_cut,min_perc=self.min_perc_test,max_perc=self.max_perc_test);
        X_train,Y_train=self.quipu_df_to_numpy(df_train);X_test,Y_test=self.quipu_df_to_numpy(df_test);
        X_train,X_valid,Y_train,Y_valid=self.divide_numpy_ds(X_train,Y_train,1-validation_prop);
        return X_train,X_valid,Y_train,Y_valid,X_test,Y_test
        
    def quipu_df_to_numpy(self,df): # dataframe data structure to numpy arrays, and barcodes in onehot encoding.
        X_numpy=np.vstack( df.trace )
        Y_barcode = np.vstack( df.barcode.values )
        Y_label=np.asarray([int(str(i[0]),2) for i in Y_barcode]);
        Y_onehot = np.zeros((Y_label.size, Y_label.max() + 1))
        Y_onehot[np.arange(Y_label.size), Y_label] = 1
        return X_numpy,Y_onehot;
    
    def divide_numpy_ds(self,X,Y,prop,keep_perc_classes=False): #Divides train in train and validation. Prop indicates proportion of train ds
    #keep perc classes assures that the classes are equally distributed in train and valid dataset.
        if (keep_perc_classes == False):
            ni_x1 = int( len(X)*prop ) # Training set
            ni_x2   = len(X) - ni_x1  # Validation set
            
            random_index = np.arange(len(X))
            np.random.shuffle(random_index) 
            
            X1=X[random_index[:ni_x1],:]
            Y1=Y[random_index[:ni_x1],:]
            X2=X[random_index[ni_x1:],:]
            Y2=Y[random_index[ni_x1:],:]
            
        else :
            print("This hasnt been implemented")
        
        return X1,X2,Y1,Y2


if __name__ == "__main__":
    dl=DataLoader();
    X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy_quipu();