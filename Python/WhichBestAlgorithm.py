import sys
from Sting import Sting
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
from crispIndices import CrispIndices
from externalIndices import ExternalIndices
from Normalization import minmax_norm
from scipy.stats import zscore 
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import listdir

if(len(sys.argv) < 3):
    print("Error usage: Introduce the algorithm, typedata: 0: all, 1: real, 2: sim")
    sys.exit()

Algorithm = sys.argv[1]
typedata = sys.argv[2]

dirpath = 'Results/IndicesResults/' + Algorithm + '/ReplacedIndices/';

dirfiles = listdir(dirpath)

Fullpandas = pd.DataFrame()

dirfiles = sorted([results for results in dirfiles if (results.split('_')[0] == 'Results' and results.split('_')[1] != 'Merged.pkl')])

for i in dirfiles:
    print(i)
    output = open(dirpath + '/' + i, "rb")
    Indices_as_pd = pickle.load(output)
    Indices_as_pd = Indices_as_pd.drop('Aditional Params', axis=1)
    print(Indices_as_pd.shape[0])
    Fullpandas = Fullpandas.append(Indices_as_pd)

Indices_as_pd = Fullpandas.reset_index(drop=True)
del Fullpandas

print('Indexing')

Indicesname = np.unique(Indices_as_pd.ix[1:100].loc[Indices_as_pd["Type"] == "Int"]["Index"])
num_internos = len(Indicesname)

Externos = np.unique(Indices_as_pd.ix[1:100].loc[Indices_as_pd["Type"] == "Ext"]["Index"])        
Externos = [x for x in Externos if x != 'Accuracy1']
num_externos = len(Externos)



# Real
Datasetreal = ['arrhythmia', 'balance', 'breast_tissue', 'ecoli', 'glass', 'haberman',
               'image', 'iono', 'iris', 'movement_libras', 'musk', 'parkinsons',
               'sonar', 'spect', 'transfusion', 'vehicle', 'vertebral', 'wdbc', 'wine',
               'winequality', 'wpbc', 'yeast', 'zoo']

Datasetsim = ['Aggregation', 'Complex 8', 'Complex 8 noise', 'Complex 9',
       'Compound', 'D31', 'Different Density', 'Dim 032', 'Dim 064',
       'Dim 128', 'Dim 256', 'Flame', 'Fourty', 'Jain', 'Pathbased', 'R15',
       'S1', 'S2', 'S3', 'S4', 'Skew clusters', 'Spiral', 'Sub clusters',
       'Well_separated', 'Well_separated noise', 't48k']

Algorithms = np.array(['Kmeans', 'EM', 'DBScan', 'Birch', 'Lamda', 'Sting'])

Datasetsname = []

if typedata == '1' or typedata == '0':
    Datasetsname = Datasetsname + Datasetreal 
if typedata == '2' or typedata == '0':
    Datasetsname = Datasetsname + Datasetsim
    
Normname = np.unique(Indices_as_pd["Norm"])

Header = Externos

print('Finished indexing')

for m in Normname:
    
    Matching = pd.DataFrame(data=np.zeros((len(Algorithms), num_externos)), index=Algorithms, columns=Header)
    
    for j in Datasetsname:
        print(m, j)
        BestExt = np.zeros(shape=(len(Externos),))
        
        ind = Indices_as_pd["Dataset"] == j
        Indicesfordataset = Indices_as_pd.loc[ind]
            
        for ext in range(len(Externos)):
            
            Indices_Externos = Indicesfordataset.loc[Indicesfordataset["Index"] == Externos[ext]]
            Indices_Externos = Indices_Externos.loc[Indices_Externos["Norm"] == m]
            Indices_Externos = Indices_Externos.loc[Indices_Externos["Clusters"] > 2]
            
            if len(Indices_Externos["Value"].values) == 0:
                    continue
            
            if Externos[ext] in ['Entropy']:
                BestExt[ext] = np.argmin(Indices_Externos["Value"])
                  
            # Max values indices  
            else:
                BestExt[ext] = np.argmax(Indices_Externos["Value"])
            
            Matching[Externos[ext]][Indicesfordataset.ix[BestExt[ext]]['Algorithm']] += 1
        
    N = len(Externos)
    ind = np.arange(N) 
    width = 0.1 
    fig = plt.figure(figsize=(15, 5), facecolor='w') 
    rects = list()
    for i, alg in enumerate(Algorithms):
        rects.append(plt.bar(ind + width * i, Matching.loc[alg, :].get_values(), width))
    plt.legend((rects), (Algorithms))
    plt.xticks(ind + width, Externos, rotation='vertical')
    plt.ylabel('Successes clustering algorithm')
    plt.savefig('Results/IndicesResults/' + Algorithm + '/Images/' + Algorithm + '_' + m + '_' + typedata + '_algorithms.png', bbox_inches='tight') 
    plt.close()
