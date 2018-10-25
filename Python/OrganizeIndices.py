import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore 
import pandas as pd


from os import listdir

if(len(sys.argv) < 2):
    print("Error usage: Introduce the algorithm")
    sys.exit()

Algorithm = sys.argv[1]

dirpath = 'Results/IndicesResults/' + Algorithm + '/ReplacedIndices/';

dirfiles = listdir(dirpath)

Fullpandas = pd.DataFrame()
dirfiles = sorted([results for results in dirfiles if results.split('_')[0] == 'Indices'])
dirfiles = sorted([results for results in dirfiles if results.split('_')[0] + '_' + results.split('_')[1] == 'Indices_for'])


for i in dirfiles:
    print(i)
    output = open(dirpath + i, "rb")
    Indices_as_pd = pickle.load(output)
    Fullpandas = Fullpandas.append(Indices_as_pd)

output = open(dirpath + "Results" + "_" + dirpath.split('/')[2] + ".pkl", 'wb')
pickle.dump(Fullpandas, output)


