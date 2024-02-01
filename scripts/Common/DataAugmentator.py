



# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:54:58 2023

@author: JK-WORK
"""
import numpy as np
from scipy import signal,interpolate
from params import QUIPU_MAGNITUDE_STD,QUIPU_STRETCH_PROB,QUIPU_STRETCH_STD,QUIPU_NOISE_AUG_STD,QUIPU_STD_FILL_DEFAULT
from DatasetFuncs import normaliseLength
import copy

import tensorflow as tf


class DataAugmentator():
    def __init__(self,brow_std=0.9,magnitude_std=QUIPU_MAGNITUDE_STD,stretch_prob=QUIPU_STRETCH_PROB,stretch_std=QUIPU_STRETCH_STD,noise_std=QUIPU_NOISE_AUG_STD,opt_aug=False): #Opt aug still has bugs, dont understand well why.
        self.stretch_std=stretch_std;
        self.magnitude_std=magnitude_std;
        self.stretch_prob=stretch_prob;
        self.noise_std=noise_std;
        self.browAug=tf.load_op_library('./../../ext/BrownianDataAugmentation/TFBrowAug/browAug.so');
        self.brow_std=brow_std;
        self.opt_aug=opt_aug;
        self.fill_noise_std=0.046;
        #self.brow_std;
    def all_augments(self,X_train):
        X = copy.deepcopy(X_train) # make copies
        if self.brow_std > 0: 
            X = self.brow_aug(X)
        if self.magnitude_std > 0:
            X = self.magnitude_aug(X, std = self.magnitude_std) 
        if self.stretch_std > 0 :
            X = self.stretch_aug(X, std=self.stretch_std, probability=self.stretch_prob) if self.opt_aug==False else self.stretch_aug_v2(X, stretch_std_val = self.stretch_std); ##Tested with profiler and stretch_aug_v2 is faster + augments all .
        if self.noise_std > 0:
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
            noise=np.random.randn(n_samples)*self.fill_noise_std;
            X_train_augmented[i,int(lengths[i]):]=noise[int(lengths[i]):] #Fills nans with the background noise instead of nans.
    

        
    ##From quipus code:
    def brow_aug(self,X_in,ret_noise=False):
        noise=np.random.randn(np.shape(X_in)[0],np.shape(X_in)[1])*self.brow_std;
        #data_out,ev_len_out= self.browAug.BrowAug(data_in=X_in,noise=noise)
        data_out_gpu,ev_len_out_gpu= self.browAug.BrowAug(data_in=X_in,noise=noise)
        #print("Memory utilization after brow Aug: " + str(tf.config.experimental.get_memory_info('GPU:0')["current"]))
        data_out=tf.identity(data_out_gpu).cpu();ev_len_out=tf.identity(ev_len_out_gpu).cpu();
        data_out=data_out.numpy();
        ev_len_out=ev_len_out.numpy();
        data_out=data_out.reshape((-1,np.shape(X_in)[1]))
        del data_out_gpu
        del ev_len_out_gpu #Removes memory used from the GPU 
        #print("Memory utilization after deletion: " + str(tf.config.experimental.get_memory_info('GPU:0')["current"]))
        self.replace_nans_w_noise(data_out,ev_len_out)
        if ret_noise:
            return data_out,noise
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
            if probability==1:
                x_new[i] = self._mutateDurationTrace(x_new[i], std)
            elif np.random.rand() > 1-probability:
                x_new[i] = self._mutateDurationTrace(x_new[i], std)
        return x_new
    def _mutateDurationTrace(self,x, std = 0.1):
        "adjust the sampling rate"
        length = len(x)
        return normaliseLength( signal.resample(x, int(length*np.random.normal(1, std))) , length = length, std_default=self.fill_noise_std)
    
    def addNoise(self,xs, std = QUIPU_NOISE_AUG_STD):
        "Add gaussian noise"
        return xs + np.random.normal(0, std, xs.shape)

    def stretch_aug_v2(self,xs, stretch_std_val = 0.08):
        x_new = np.copy(xs)
        n_reads=np.shape(xs)[1];
        length=np.shape(xs)[0];
        new_lens=(length*np.random.normal(1, stretch_std_val,n_reads)).astype(int);
        n_noise_needed=np.sum(length-new_lens[new_lens<length]); ##New samples of noise needed to fill
        fill_noise=np.random.normal(0, QUIPU_STD_FILL_DEFAULT,n_noise_needed);
        count_noise=0;
        for i in range(n_reads):
            f = interpolate.interp1d(np.arange(length), xs[:,i])
            if new_lens[i]<length:
                n_samples_fill=length-new_lens[i];
                x_new[:new_lens[i],i]=f(np.arange(new_lens[i]));
                x_new[new_lens[i]:,i]=fill_noise[count_noise:count_noise+n_samples_fill];
                count_noise=count_noise+n_samples_fill;
            else:
                x_idx_interpol=np.linspace(0,(length/new_lens[i])*(length-1),num=length);
                x_new[:,i]=f(x_idx_interpol);
            
        return x_new

    

def test_brow_aug(X_train,out_aug):
    import matplotlib.pyplot as plt
    
    for i in range(10):
        plt.plot(X_train[i,:])
        plt.plot(out_aug[i,:])
        plt.show()

def test_stretch_v2():
    import matplotlib.pyplot as plt
    idxs=np.arange(700);
    n_runs=10;
    x=1/np.abs(idxs-350.5);
    x=np.tile(x,(n_runs,1)).T
    da=DataAugmentator();
    out=da.stretch_aug_v2(x);
    plt.figure();
    for i in range(n_runs):
        plt.plot(out[:,i])
    plt.show()


def test_weird_noise(): #It happened that in some aug samples a different noise appears on the right side. 
    from DataLoader import DataLoader
    import matplotlib.pyplot as plt
    dl=DataLoader();
    np.random.seed(1);
    da=DataAugmentator();
    X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy();
    out=da.brow_aug(X_train[:100,:]);
    for i in range(20):
        plt.plot(X_train[i,:],label="Original")
        plt.plot(out[i,:],label="Aug")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    test_weird_noise()
    # from DataLoader import DataLoader
    # dl=DataLoader();
    # #X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy_quipu();
    # X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy();
    # da=DataAugmentator();
    # out=da.brow_aug(X_train);
    # test_brow_aug(X_train,out);
