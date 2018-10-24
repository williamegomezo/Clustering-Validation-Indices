# Python Packages
import sys
import numpy as np
import pickle
import timeit
import pandas as pd
import timeit
import os

# Indices Package
from Library.crispIndices import CrispIndices
from Library.externalIndices import ExternalIndices

# Standarizations
from Utils.Normalization import minmax_norm
from scipy.stats import zscore 

if(len(sys.argv) < 4):
    print("Error usage: Introduce the algorithm name and range from which dataset start computing indices (1:) and to when to stop")
    sys.exit()
    
Algorithm = sys.argv[1]
Inicio = int(sys.argv[2]) - 1
Fin = int(sys.argv[3])

output = open("Datasets/Datasets.pkl", "rb")
Datasets = pickle.load(output)

dirpath = 'Results/ResultsClustering/' + Algorithm + '/'
dirpath2 = 'Results/IndicesResults/' + Algorithm + '/'
dirfiles = os.listdir(dirpath)

dirfiles = sorted([dataset for dataset in dirfiles if dataset.split('_')[1] != 'Birch 1'])

if Fin > len(dirfiles):
    Fin = len(dirfiles)

for i in range(Inicio, Fin):
    Indices_as_pd = pd.DataFrame(columns=['Index', 'Type', 'Dataset', 'Algorithm', 'Clusters', 'Norm', 'Aditional Params', 'Value', 'Time'])
    
    output = open(dirpath + dirfiles[i], "rb")
    ClusteringResults = pickle.load(output)
    
    for j in range(len(ClusteringResults)):
        print(Algorithm, i, dirfiles[i], j, len(ClusteringResults))
        
        index = np.where(ClusteringResults.loc[j]['Title'] == Datasets["Title"])[0]
        
        data = Datasets.iloc[index]["Data"].values[0]
        label = Datasets.loc[index]["Labels"].values[0]
        recon = ClusteringResults.loc[j]['Recon'].astype(int)
        Normname = ClusteringResults.loc[j]['Norm']
        
        if np.sum(recon == -1) > 0:  # outliers, considered as just one-element clusters
            ind = np.where(recon == -1)[0]
            maximo = np.max(recon)
            for l in range(len(ind)):
                recon[ind[l]] = maximo + l + 1
                
        if np.sum(label == -1) > 0:  # outliers, considered as just one-element clusters
            ind = np.where(label == -1)[0]
            maximo = np.max(label)
            for l in range(len(ind)):
                label[ind[l]] = maximo + l + 1
        
        howmany = np.max(recon)
        p = 1
        for r in range(1, howmany + 1):
            if np.sum(p == recon) == 0:
                recon[recon > p] = recon[recon > p] - 1
            else:
                p += 1
            
        
        if np.max(recon) < 2 or np.max(recon) > 4 * np.max(label):
            continue
        
        if Normname == "No norm":
            datan = data
            
        if Normname == "Min-Max":
            datan = minmax_norm(data)
            
        if Normname == "Zscore":
            datan = zscore(data, axis=0)
        
        CVE = ExternalIndices(label, recon.astype(int))
        CVI = CrispIndices(datan, recon.astype(int))

        ListofInternals = [method for method in dir(CVI) if callable(getattr(CVI, method)) and method.split('_')[0] == 'Int']
        ListofExternals = [method for method in dir(CVE) if callable(getattr(CVE, method)) and method.split('_')[0] == 'Ext']
        
        spm = False
        if data.shape[0] < 500:
            spm = True
            s_plus, s_minus = CVI._s_plus_minus()
        
        for m in range(len(ListofInternals)):
            
            
            Indexmethod = ListofInternals[m]
            Indexname = Indexmethod.replace('_', ' ')[4:]
            print(Algorithm, i , dirfiles[i], j, len(ClusteringResults), m, len(ListofInternals), Indexname)
            
            
            start_time = timeit.default_timer()
            if Indexmethod in ['Int_ARsd']:
                alpha = 0.01
                Indexvalue = getattr(CVI, Indexmethod)(alpha)
            
            elif Indexmethod in ['Int_Baker_Hubert_Gamma', 'Int_Tau']:
                if spm:
                    Indexvalue = getattr(CVI, Indexmethod)(s_plus, s_minus)
                else:
                    Indexvalue = np.nan
                
            elif Indexmethod in ['Int_G_Plus']:    
                if spm:
                    Indexvalue = getattr(CVI, Indexmethod)(s_minus)
                else:
                    Indexvalue = np.nan  
            else:
                Indexvalue = getattr(CVI, Indexmethod)()
            Time = timeit.default_timer() - start_time   
                
            row = Indices_as_pd.shape[0]
            Indices_as_pd.set_value(row, 'Index', Indexname)
            Indices_as_pd.set_value(row, 'Type', 'Int')
            Indices_as_pd.set_value(row, 'Dataset', ClusteringResults.loc[j]['Title'])
            Indices_as_pd.set_value(row, 'Algorithm', Algorithm)
            Indices_as_pd.set_value(row, 'Clusters', ClusteringResults.loc[j]['Clusters'])
            Indices_as_pd.set_value(row, 'Norm', Normname) 
            Indices_as_pd.set_value(row, 'Aditional Params', ClusteringResults.loc[j]['Aditional Params']) 
            Indices_as_pd.set_value(row, 'Value', Indexvalue) 
            Indices_as_pd.set_value(row, 'Time', Time) 
            
            
            output = open(dirpath2 + 'Indices_for_' + Algorithm + '_' + ClusteringResults.loc[j]['Title'] + '_' + Normname + '.pkl', 'wb')
            pickle.dump(Indices_as_pd, output)
        
        for m in range(len(ListofExternals)):
            
            Indexmethod = ListofExternals[m]
            Indexname = Indexmethod.replace('_', ' ')[4:]
            print(Algorithm, i, dirfiles[i], j, len(ClusteringResults), m, len(ListofExternals), Indexname)
            
            start_time = timeit.default_timer()
            Indexvalue = getattr(CVE, Indexmethod)()
            Time = timeit.default_timer() - start_time   
                
                
            Indexname = Indexmethod.replace('_', ' ')[4:]

            row = Indices_as_pd.shape[0]
            Indices_as_pd.set_value(row, 'Index', Indexname)
            Indices_as_pd.set_value(row, 'Type', 'Ext')
            Indices_as_pd.set_value(row, 'Dataset', ClusteringResults.loc[j]['Title'])
            Indices_as_pd.set_value(row, 'Algorithm', Algorithm)
            Indices_as_pd.set_value(row, 'Clusters', ClusteringResults.loc[j]['Clusters'])
            Indices_as_pd.set_value(row, 'Norm', Normname) 
            Indices_as_pd.set_value(row, 'Aditional Params', ClusteringResults.loc[j]['Aditional Params']) 
            Indices_as_pd.set_value(row, 'Value', Indexvalue) 
            Indices_as_pd.set_value(row, 'Time', Time) 
            
            
            output = open(dirpath2 + 'Indices_for_' + Algorithm + '_' + ClusteringResults.loc[j]['Title'] + '_' + Normname + '.pkl', 'wb')
            pickle.dump(Indices_as_pd, output)
    
    output = open(dirpath2 + 'Indices_for_' + Algorithm + '_' + ClusteringResults.loc[j]['Title'] + '_' + Normname + 'FULL.pkl', 'wb')
    pickle.dump(Indices_as_pd, output)
    
    if os.path.exists(dirpath2 + 'Indices_for_' + Algorithm + '_' + ClusteringResults.loc[j]['Title'] + '_' + Normname + '.pkl'):
        os.remove(dirpath2 + 'Indices_for_' + Algorithm + '_' + ClusteringResults.loc[j]['Title'] + '_' + Normname + '.pkl')
        