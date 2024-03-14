#import cupy as cp
#import cupyx.scipy as csc
#import inspect

#print([o[0] for o in inspect.getmembers(csc) if inspect.ismodule(o[1])])


#import cupy as cp
#from cupyx.scipy.interpolate import interpn
#from cupyx.scipy.signal import resample

#def value_func_3d(x, y, z):
#    return 2 * x + 3 * y - z
#x = cp.linspace(0, 4, 5)
#y = cp.linspace(0, 5, 6)
#z = cp.linspace(0, 6, 7)
#points = (x, y, z)
#values = value_func_3d(*cp.meshgrid(*points, indexing='ij'))

import numpy as np
from scipy import signal,interpolate
import time


##Augmentations from quipu

def noiseAug(xs, std = 0.1):#Noise augmentation
    return xs + np.random.normal(0, std, xs.shape)

def magnitudeAug(xs, std = 0.1):#Magnitude augmentation
    return xs * np.abs(np.random.normal(1, std, len(xs)).reshape((-1,1)) ) 

def stretchAug(xs, std = 0.1, probability = 0.5,std_fill=0.08):#Stretch augmentation
    x_new = np.copy(xs)
    sample_len=len(x_new[1])
    for i in range(len(xs)):
        if np.random.rand() > 1-probability:
            x_resampled=signal.resample(x_new[i], int(sample_len*np.random.normal(1, std))) 
            if len(x_resampled)<700:
                x_resampled= np.append(x_resampled,np.random.normal(0, std_fill, sample_len - len(x_resampled)))
            else:
                x_resampled=x_resampled[:700]
            x_new[i] = x_resampled;
    return x_new


##Augmentations with cupy
import cupy as cp
import cupyx

def noiseAug_gpu(xs, std = 0.1):#Noise augmentation
    return xs + cp.random.normal(0, std, xs.shape)

def magnitudeAug_gpu(xs, std = 0.1):
    return xs * cp.abs(cp.random.normal(1, std, len(xs)).reshape((-1,1)) ) 

def stretchAug_gpu(xs, std = 0.1,std_fill=0.01, probability = 0.5,):
    x_new = cp.zeros(cp.shape(xs));
    sample_len=len(x_new[1])
    step_sizes=cp.random.normal(1,std,size=(len(xs)));
    min_len=int(cp.min(step_sizes*sample_len));
    x_new[:,min_len:]=cp.random.normal(0,std_fill,size=(len(xs),sample_len-min_len));

    for i in range(len(xs)):
        x_new_idxs=step_sizes[i]*cp.arange(sample_len);
        x_resampled=cp.interp(x_new_idxs[x_new_idxs<sample_len],cp.arange(sample_len),xs[i])
        x_new[i,:len(x_resampled)]=x_resampled;
    return x_new

def stretchAug_gpu_2(xs, std = 0.1,std_fill=10, probability = 0.5,):
    n_samples=int(len(xs));sample_len=int(len(xs[0]));
    #x_new = np.zeros(np.shape(xs));
    step_sizes=cp.random.normal(1,std,size=(n_samples,1));
    new_lens=(step_sizes*sample_len).astype(int)
    min_len=int(cp.min(new_lens));
    fill_noise=cp.random.normal(0,std_fill,size=(n_samples,sample_len-min_len));
    base_idxs=cp.arange(sample_len).reshape(1,sample_len)
    interpol_idxs=cp.matmul(step_sizes,base_idxs); #Matrix containing in each row the interpolating x
    samples_to_exclude_mask=interpol_idxs>699; ##Samples that have x that we dont have are excluded
    interpol_idxs[samples_to_exclude_mask]=0; #Interpolating indexes above 699 are set to 0
    interpol_idxs_floor=cp.floor(interpol_idxs).astype(int); #Floor of interpolating indexes

    row_indicator=cp.tile(cp.reshape(np.arange(n_samples),(n_samples,1)),(1,sample_len)) #Matrix where each element is the row index
    x_base=xs[row_indicator,interpol_idxs_floor]; #using the floor of the interpolating indexes to get the the value for the given indexes

    multiplier= interpol_idxs-interpol_idxs_floor; #Interpolating values between 0 and 1 (n_samples,1)
    difference=xs[row_indicator,interpol_idxs_floor+1]-x_base; #Difference between the Y(x) where x is the floor and floor + 1 of the interpolated ones

    x_interpolated = x_base + cp.multiply(multiplier, difference) #Linear difference.

    #x_interpolated_sub_ending=x_interpolated[:,min_len:];
    x_interpolated[samples_to_exclude_mask]=fill_noise[samples_to_exclude_mask[:,min_len:]];
    #x_interpolated[samples_to_exclude_mask[:,min_len:]]=fill_noise[samples_to_exclude_mask[:,min_len:]]; ##Fills with noise the samples that were excluded

    return x_interpolated

trace_to_use_example=np.arange(700); ##Just a trace to 
xs=np.tile(trace_to_use_example, (60000, 1));
xs_gpu=cp.array(xs)
def timeAndTestFnc(xs,fnc, nameFnc=""):
    start = time.time()
    fnc(xs)
    end = time.time()
    print("Time of "+ nameFnc + ": "+ str(end - start))

#timeAndTestFnc(xs,noiseAug, nameFnc="CPU Noise Aug")
#timeAndTestFnc(xs,magnitudeAug, nameFnc="CPU Mag Aug")
#timeAndTestFnc(xs,stretchAug, nameFnc="CPU Stretch Aug")

#timeAndTestFnc(xs_gpu,noiseAug_gpu, nameFnc="GPU Noise Aug")
#timeAndTestFnc(xs_gpu,magnitudeAug_gpu, nameFnc="GPU Mag Aug")
timeAndTestFnc(xs_gpu,stretchAug_gpu_2, nameFnc="GPU Stretch Aug")
