# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:54:58 2023

@author: JK-WORK
"""
import numpy as np
from scipy import signal
from params import QUIPU_MAGNITUDE_STD,QUIPU_STRETCH_PROB,QUIPU_STRETCH_STD
from DatasetFuncs import normaliseLength
import copy


class DataAugmentator():
    def __init__(self,brow_std=1,magnitude_std=0.08,stretch_prob=0.5,stretch_std=0.1):
        self.stretch_std=stretch_std;
        self.magnitude_std=magnitude_std;
        self.stretch_prob=stretch_prob;
        #self.brow_std;
    def quipu_augment(self,X_train):
        X = copy.deepcopy(X_train) # make copies
        X = self.magnitude_aug(X, std = self.magnitude_std) 
        X = self.stretch_aug(X, std=self.stretch_std, probability=self.stretch_prob)
        return X;
    
    
    ##From quipus code:
    def magnitude_aug(self,xs, std = 0.15):
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
    
    def addNoise(self,xs, std = 0.05):
        "Add gaussian noise"
        return xs + np.random.normal(0, std, xs.shape)
