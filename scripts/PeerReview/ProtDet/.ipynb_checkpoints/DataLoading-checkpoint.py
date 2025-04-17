import pandas as pd
import numpy as np
import os
from params import QUIPU_DATA_FOLDER,QUIPU_LEN_CUT,QUIPU_STD_FILL_DEFAULT


def get_datasets_numpy(df_train,df_test,validation_prop=0.05,repeat_classes=True):
    
    X_train,Y_train=quipu_df_to_numpy(df_train);X_test,Y_test=quipu_df_to_numpy(df_test);
    X_train,X_valid,Y_train,Y_valid=divide_numpy_ds(X_train,Y_train,1-validation_prop,keep_perc_classes=True,repeat_classes=repeat_classes);
    return X_train,X_valid,Y_train,Y_valid,X_test,Y_test

def quipu_df_to_numpy(df): # dataframe data structure to numpy arrays, and barcodes in onehot encoding.
    X_numpy=np.vstack( df.trace )
    Y = np.vstack( df.Bound.values ) #1D binary signal in this case
    return X_numpy,Y; 

def divide_numpy_ds(X,Y,prop,keep_perc_classes=False,repeat_classes=False): #Divides train in train and validation. Prop indicates proportion of train ds
    #keep perc classes assures that the classes are equally percentally distributed in train and valid dataset.
    #repeat_classes repeats reads so each class has the same amount of samples
    ni_x1 = int( len(X)*prop ) # Training set length
    ni_x2   = len(X) - ni_x1  # Validation set length
    if (keep_perc_classes == False):
        indexes_to_partition = np.arange(len(X))
        np.random.shuffle(indexes_to_partition)  # Shuffles indexes of random
        X1=X[indexes_to_partition[:ni_x1],:] #Based on the random indexes picks the datasets
        Y1=Y[indexes_to_partition[:ni_x1],:]
        X2=X[indexes_to_partition[ni_x1:],:]
        Y2=Y[indexes_to_partition[ni_x1:],:]
    else :
        idxs_train=[];idxs_valid=[]
        #ipdb.set_trace(); 
        ids_present_antibody=np.argwhere(Y==1)[:,0] ##Only to check classes
        ids_not_present_antibody=np.argwhere(Y==0)[:,0] ##Only to check classes

        n_train_pa=int(prop*len(ids_present_antibody));
        n_train_npa=int(prop*len(ids_not_present_antibody));

        np.random.shuffle(ids_present_antibody)
        np.random.shuffle(ids_not_present_antibody)

        idxs_train.append(ids_present_antibody[:n_train_pa]);idxs_valid.append(ids_present_antibody[n_train_pa:])
        idxs_train.append(ids_not_present_antibody[:n_train_npa]);idxs_valid.append(ids_not_present_antibody[n_train_npa:])

        if repeat_classes:
            idxs_train=repeat_reads_to_balance(idxs_train)
        idxs_train=np.concatenate(idxs_train);idxs_valid=np.concatenate(idxs_valid); #List of lists to numpy array
        X1=X[idxs_train,:];Y1=Y[idxs_train,:];
        X2=X[idxs_valid,:];Y2=Y[idxs_valid,:];
    
    return X1,X2,Y1,Y2

def repeat_reads_to_balance(idxs_train_in):
    n_evs_per_class = np.asarray([len(i) for i in idxs_train_in]);
    n_evs_goal = np.max(n_evs_per_class);
    idxs_train_out=[np.resize(i,(n_evs_goal,)) for i in idxs_train_in]
    return idxs_train_out;

def get_dataset_as_Quipu(data_folder):
    df_cut=allDataset_loader(data_folder,cut=True)
    df_train,df_test=dataset_split_as_quipu(df_cut)
    return df_train,df_test

def normaliseLength(trace, length = QUIPU_LEN_CUT, trim = 0, std_default=QUIPU_STD_FILL_DEFAULT): ##Paramters given in quipus code
    """
    Normalizes the length of the trace and trims the front 
    
    :param length: length to fit the trace into (default: 600)
    :param trim: how many points to drop in front of the trace (default: 0)
    :return: trace of length 'length' 
    """
    if len(trace) >= length + trim:
        return trace[trim : length+trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, std_default, length - len(trace[trim:]))
        )    

def allDataset_loader(data_folder,path_dataset_preprocessed="../../../data/datasetBound.hdf5",cut=True):
    df_name='datasetQuipu' if cut else 'datasetQuipuUncut'
    if os.path.exists(path_dataset_preprocessed):
        allDatasets=pd.read_hdf(path_dataset_preprocessed, df_name)  
    else:
        #Datasets selected for Quipu training in the original script
        dataset =         pd.concat([ 
            pd.read_hdf(data_folder+"dataset_part1.hdf5"),
            pd.read_hdf(data_folder+"dataset_part2.hdf5")
        ])
        #datasetTestEven = pd.read_hdf(data_folder+"datasetTestEven.hdf5")
        #datasetTestOdd =  pd.read_hdf(data_folder+"datasetTestOdd.hdf5")
        #datasetTestMix =  pd.read_hdf(data_folder+"datasetTestMix.hdf5")
        datasetWithAntibodies =  pd.concat([ 
            pd.read_hdf(data_folder+"datasetWithAntibodies_part1.hdf5"),
            pd.read_hdf(data_folder+"datasetWithAntibodies_part2.hdf5")
        ])
        datasetExtra =    pd.read_hdf(data_folder+"datasetExtra.hdf5")
        
        allDatasets = pd.concat([dataset , datasetExtra, datasetWithAntibodies],ignore_index = True)
        allDatasets = allDatasets[allDatasets.Filter] # clear bad points
    
        #Here I should normalize the traces
        traces = allDatasets.trace
        traces_uniform = traces.apply(lambda x: normaliseLength(x))
        traces_normalised_unif =  - traces_uniform / allDatasets.UnfoldedLevel 
        traces_normalised = - traces / allDatasets.UnfoldedLevel;
        allDatasets.trace = traces_normalised_unif;
        quipu_ds=allDatasets[ ["Bound","barcode", "nanopore","trace"]]; #Keep the only information that we will use
        quipu_ds.to_hdf(path_dataset_preprocessed, 'datasetQuipu');
        allDatasets.trace = traces_normalised;
        uncut_ds=allDatasets[ ["Bound", "barcode","nanopore","trace"]]; #Keep the only information that we will use
        uncut_ds.to_hdf(path_dataset_preprocessed, mode='a', key='datasetQuipuUncut');
        allDatasets=quipu_ds if cut else uncut_ds;
    return allDatasets;


def dataset_split_as_quipu(allDatasets): 
    
    # final selection: different from previous one - use this only after picking the final model
    testSetIndex = [
        # barcode, nanopore
        ('000', 6),
        ('001', 26),
        ('010', 1159),
        ('011', 35),  # unbound
        ('011', 32),  # bound
        ('100', 1933),
        ('101', 30),
        ('110', 12),
        ('111', 14)
    ]

    
    testSetSelection = allDatasets[["barcode", "nanopore"]]\
                            .apply(tuple, axis = 1)\
                            .isin(testSetIndex)
    
    testSet = allDatasets[ testSetSelection ]
    trainSet = allDatasets[ ~ testSetSelection ]
    
    return trainSet,testSet
