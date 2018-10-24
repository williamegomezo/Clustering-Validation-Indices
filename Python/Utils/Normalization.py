import numpy as np

def minmax_norm(data):
    Mindata = np.tile(np.min(data,axis=0),(data.shape[0],1));
    Maxdata = np.tile(np.max(data,axis=0),(data.shape[0],1));
    return (data-Mindata)/(Maxdata-Mindata)