# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:54:58 2023

@author: JK-WORK
"""
import numpy as np
from scipy import signal
from params import QUIPU_MAGNITUDE_STD,QUIPU_STRETCH_PROB,QUIPU_STRETCH_STD,QUIPU_NOISE_AUG_STD,QUIPU_STD_FILL_DEFAULT
from DatasetFuncs import normaliseLength
import copy

import tensorflow as tf


class DataAugmentator():
    def __init__(self,brow_std=0.9,magnitude_std=QUIPU_MAGNITUDE_STD,stretch_prob=QUIPU_STRETCH_PROB,stretch_std=QUIPU_STRETCH_STD,noise_std=QUIPU_NOISE_AUG_STD):
        self.stretch_std=stretch_std;
        self.magnitude_std=magnitude_std;
        self.stretch_prob=stretch_prob;
        self.noise_std=noise_std;
        self.browAug=tf.load_op_library('./../ext/BrownianDataAugmentation/TFBrowAug/browAug.so');
        self.brow_std=brow_std;
        #self.brow_std;
    def all_augments(self,X_train):
        X = copy.deepcopy(X_train) # make copies
        X = self.brow_aug(X)
        X = self.magnitude_aug(X, std = self.magnitude_std) 
        X = self.stretch_aug(X, std=self.stretch_std, probability=self.stretch_prob)
        X = self.addNoise( X, std = self.noise_std) 
        return X;
    def quipu_augment(self,X_train):
        X = copy.deepcopy(X_train) # make copies
        X = self.magnitude_aug(X, std = self.magnitude_std) 
        X = self.stretch_aug(X, std=self.stretch_std, probability=self.stretch_prob)
        X = self.addNoise( X, std = self.noise_std) 
        return X;
    
    def replace_nans_w_noise(self,X_train_augmented,lengths):
        n_samples=np.shape(X_train_augmented)[1];
        idxs_to_fix=np.argwhere(lengths<n_samples)[:,0]
        for i in idxs_to_fix:
            noise=np.random.randn(n_samples)*QUIPU_STD_FILL_DEFAULT;
            X_train_augmented[i,int(lengths[i]):]=noise[int(lengths[i]):] #Fills nans with the background noise instead of nans.
    

        
    ##From quipus code:
    def brow_aug(self,X_in):
        noise=np.random.randn(np.shape(X_in)[0],np.shape(X_in)[1])*self.brow_std;
        data_out,ev_len_out= self.browAug.BrowAug(data_in=X_in,noise=noise)
        data_out=data_out.numpy();
        ev_len_out=ev_len_out.numpy();
        data_out=data_out.reshape((-1,np.shape(X_in)[1]))
        self.replace_nans_w_noise(data_out,ev_len_out)
        return data_out
    
    def magnitude_aug(self,xs, std = QUIPU_MAGNITUDE_STD):
        "Baseline mutation"
        return xs * np.abs(np.random.normal(1, std, len(xs)).reshape((-1,1)) ) 
    def stretch_aug(self,xs, std = 0.1, probability = 0.5):
        """
        Augment the length by re-sampling. probability gives ratio of mutations
        Slow method since it uses scipy
        
        :param xs: input numpy data structure
        :param std: amound to mutate (drawn from normal distribution)
        :param probability: probabi
        """
        x_new = np.copy(xs)
        for i in range(len(xs)):
            if np.random.rand() > probability:
                x_new[i] = self._mutateDurationTrace(x_new[i], std)
        return x_new
    def _mutateDurationTrace(self,x, std = 0.1):
        "adjust the sampling rate"
        length = len(x)
        return normaliseLength( signal.resample(x, int(length*np.random.normal(1, std))) , length = length)
    
    def addNoise(self,xs, std = QUIPU_NOISE_AUG_STD):
        "Add gaussian noise"
        return xs + np.random.normal(0, std, xs.shape)

    

def test_brow_aug(X_train,out_aug):
    import matplotlib.pyplot as plt
    
    for i in range(10):
        plt.plot(X_train[i,:])
        plt.plot(out_aug[i,:])
        plt.show()


if __name__ == "__main__":
    from DataLoader import DataLoader
    dl=DataLoader();
    #X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy_quipu();
    X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy();
    da=DataAugmentator();
    out=da.brow_aug(X_train);
    test_brow_aug(X_train,out);