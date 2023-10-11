import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from DataLoader import DataLoader;
from DataAugmentator import DataAugmentator;
import cProfile
import pstats


dl=DataLoader();
da=DataAugmentator(brow_std=0.9,opt_aug=False);
X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy();

cProfile.run('a=da.all_augments(X_train)', 'restats')
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(20)

    #cProfile.run('a=da.all_augments(X_train)', 'restats')
    #p = pstats.Stats('restats')
    #p.sort_stats('cumulative').print_stats(10)