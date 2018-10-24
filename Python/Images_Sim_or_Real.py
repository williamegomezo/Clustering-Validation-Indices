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
from os import mkdir

if(len(sys.argv) < 3):
    print("Error usage: Introduce the algorithm, typedata: 0: all, 1: real, 2: sim")
    sys.exit()

Algorithm = sys.argv[1]
typedata = sys.argv[2]
try:
    mkdir('Results/IndicesResults/' + Algorithm + '/Images/')
except:
    pass
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

Datasetsname = []

if typedata == '1':
    Datasetsname = [Datasetreal]
if typedata == '2':
    Datasetsname = [Datasetsim]
if typedata == '0':  
    Datasetsname = [Datasetreal, Datasetsim]
    
Normname = np.unique(Indices_as_pd["Norm"])

Header = Externos

print('Finished indexing')

for m in Normname:
    
    try:
        mkdir('Results/IndicesResults/' + Algorithm + '/Images/' + m)
    except:
        pass
    
    MatchingMatriz = list()
    
    for k in range(len(Datasetsname)):
        
        Matching = pd.DataFrame(data=np.zeros((num_internos, num_externos)), columns=Header)
        Matching["Index"] = [i for i in Indicesname]
        for j in Datasetsname[k]:
            print(m, j)
            BestExt = np.zeros(shape=(len(Externos),))
            
            ind = Indices_as_pd["Dataset"] == j
            print(np.sum(ind))
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
                    
            
            
            p = 0
            for i in range(len(Indicesname)):
             
                # print(Indicesname[i])
                Indices_Results_for_Dataset = Indicesfordataset.loc[Indicesfordataset["Index"] == Indicesname[i]]
                
                Indices_Results_for_Dataset = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Norm"] == m]
                
                Indices_Results_for_Dataset = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] > 2]
                
                
                if (not(Indicesname[i] == "CNN 005" or Indicesname[i] == "CNN 010" or Indicesname[i] == "CNN 020" or Indicesname[i] == "CNN 050" or Indicesname[i] == "CNN 100")):
                    if np.array(Indices_Results_for_Dataset["Value"].get_values().tolist()).ndim == 1:
                        Indnan = np.isnan(Indices_Results_for_Dataset["Value"].get_values().astype(float))
                    
                        Indices_Results_for_Dataset = Indices_Results_for_Dataset.drop(Indices_Results_for_Dataset.index[Indnan])
                
                
                if len(Indices_Results_for_Dataset["Value"].values) == 0:
                    p += 1
                    continue
                
                # Min values indices
                if Indicesname[i] in ["Banfeld Raftery", "C index",
                         "Davies Bouldin", "G Plus", "Mcclain Rao", "Ray Turi",
                         "Scott Symons", "Sdbw", "Xie Beni", 'COP', 'CS Measure',
                         'Negentropy Increment Biased', 'Negentropy Increment C',
                         'SymDB']:
                    
                    
                    
                    Bestind = np.argmin(Indices_Results_for_Dataset["Value"])
                    ((Bestind < BestExt) * (BestExt < Bestind + num_internos)).astype(int)
                    
                    # print(i)
                    
                # Max values indices  
                elif Indicesname[i] in ["AIC", "BIC", "Calinski Harabasz", 'Dunn11',
                           'Dunn12', 'Dunn13', 'Dunn14', 'Dunn15', 'Dunn16', 'Dunn21',
                           'Dunn22', 'Dunn23', 'Dunn24', 'Dunn25', 'Dunn26', 'Dunn31',
                           'Dunn32', 'Dunn33', 'Dunn34', 'Dunn35', 'Dunn36',
                           "Baker Hubert Gamma", 'PBM', 'Point Biserial',
                           'Ratkowsky Lance', 'Silhouette', 'Tau', 'Wemmert Gancarski',
                           'Score function', 'PC', 'EC', 'CVM', 'ARsd', 'Sym',
                           'SymD', 'Sym33', 'SV', 'OS']:
                    
                    
                
                    Bestind = np.argmax(Indices_Results_for_Dataset["Value"])
                    ((Bestind < BestExt) * (BestExt < Bestind + num_internos + num_externos)).astype(int)
                
                    # print(i)
                
                elif Indicesname[i] in ['Ball Hall', 'Ksq Det W', 'Trace W', 'Trace WIB', 'Det Ratio', 'Log Det Ratio', 'Log SS Ratio', 'WB Index', 'WB Index zhao', 'Xu Index']:
                    
                    
                    nclusters = np.unique(Indices_Results_for_Dataset["Clusters"])
                    optimizationlike = np.zeros(shape=nclusters.shape)
                 
                    if len(nclusters) < 3:
                        p += 1
                        continue
                     
                    for k in range(len(nclusters)):
                        optimizationlike[k] = np.mean(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == nclusters[k]]["Value"])
                    
                    x = nclusters[1:-1]
                    Indice = np.abs(np.diff(optimizationlike, 1)[:-1] / np.roll(np.diff(optimizationlike, 1), -1)[:-1])
                 
                    Bestk = x[np.argmax(Indice)]
    
                    Bestindices = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk]
                    indbe = np.argmin(np.abs(Bestindices["Value"].values - optimizationlike[nclusters == Bestk]))
    
                    Bestind = Bestindices.index.values[indbe]
                    
                    # print(i)
                    
                elif Indicesname[i] in ['STR']:
                    nclusters = np.unique(Indices_Results_for_Dataset["Clusters"])
                    E = np.zeros(shape=(nclusters.shape))
                    D = np.zeros(shape=(nclusters.shape))
                    STR = np.zeros(shape=(nclusters.shape))
                 
                    if len(nclusters) < 3:
                        p += 1
                        continue
                     
                    for k in range(len(nclusters)):
                        Indices_for_k = np.array(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == nclusters[k]]["Value"].values.tolist())
                        E[k], D[k] = np.mean(Indices_for_k, axis=0)
                         
                    for k in range(0, len(nclusters) - 2):
                        STR[k] = (E[k + 1] - E[k]) * (D[k + 2] - D[k + 1])
                        
                    Bestk = nclusters[np.argmax(STR)]
                    
                    Bestindices = np.argmin(np.linalg.norm(np.array(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk]["Value"].values.tolist()) - np.array([E[np.argmax(STR)], D[np.argmax(STR)]]), axis=1))
    
                    Bestind = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk].index.values[Bestindices]
                     
                    # print(i)
                 
                elif Indicesname[i] in ['SD']:
                    nclusters = np.unique(Indices_Results_for_Dataset["Clusters"])
                    S = np.zeros(shape=(nclusters.shape))
                    D = np.zeros(shape=(nclusters.shape))
                    SD = np.zeros(shape=(nclusters.shape))
                 
                     
                     
                    for k in range(len(nclusters)):
                        Indices_for_k = np.array(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == nclusters[k]]["Value"].values.tolist())
                        D[k], S[k] = np.mean(Indices_for_k, axis=0)
                         
                    SD = D[-1] * S + D
                    
                    Bestk = nclusters[np.argmin(SD)]
                    
                    Bestindices = np.argmin(np.linalg.norm(np.array(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk]["Value"].values.tolist()) - np.array([S[np.argmin(SD)], D[np.argmin(SD)]]), axis=1))
    
                    Bestind = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk].index.values[Bestindices]
                     
                    # print(i)
                    
                elif Indicesname[i] in ['Variance of Negentropy']:
                    MeanNegentropy = Indicesfordataset.loc[Indicesfordataset["Index"] == "Negentropy Increment C"]
                    MeanNegentropy = MeanNegentropy.loc[MeanNegentropy["Norm"] == m]
                     
                    Meaneg = np.zeros(shape=(nclusters.shape))
                    Stdneg = np.zeros(shape=(nclusters.shape))
                     
                    nclusters = np.unique(Indices_Results_for_Dataset["Clusters"])
                 
                    for k in range(len(nclusters)):
                        Meaneg[k] = np.mean(MeanNegentropy.loc[MeanNegentropy["Clusters"] == nclusters[k]]["Value"])
                        Stdneg[k] = np.mean(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == nclusters[k]]["Value"])
                    
                    
                    
                    Pmin = np.argmin(Meaneg)
                    Ind = np.where(Meaneg - Stdneg <= Meaneg[Pmin] + Stdneg[Pmin])[0]  
                    if len(Ind) > 0:
                        LowerStd = np.argmin(Stdneg[Ind])
                        Pmin = Ind[LowerStd]
                        
                    Bestk = nclusters[Pmin]
                    
                    Bestindices = MeanNegentropy.loc[MeanNegentropy["Clusters"] == Bestk]
                    indbe = np.argmin(np.abs(Bestindices["Value"].values - Meaneg[Pmin]))
    
                    Bestind = Bestindices.index.values[indbe]
                    
                    # print(i)
                     
                elif Indicesname[i] in ['Bhargavi Gowda']:  # Cut-Off Rule
                                        
                    Bhargavi = np.zeros(shape=(nclusters.shape))
                     
                    nclusters = np.unique(Indices_Results_for_Dataset["Clusters"])
                    
                    if len(nclusters) < 3:
                        p += 1
                        continue
                     
                    for k in range(len(nclusters)):
                        Bhargavi[k] = np.mean(Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == nclusters[k]]["Value"])
                     
                    BG = np.roll(Bhargavi, -1)[:-1] / Bhargavi[:-1]
                     
                    if BG.shape[0] == 0:
                        p += 1
                        continue
                    
                    BG[BG > 1] = 1 / BG[BG > 1] 
                     
                    BG[np.isnan(BG)] = 0 
                     
                    Bestk = nclusters[np.argmin(BG)]
                     
                    Bestindices = Indices_Results_for_Dataset.loc[Indices_Results_for_Dataset["Clusters"] == Bestk]
                    indbe = np.argmin(np.abs(Bestindices["Value"].values - Bhargavi[np.argmin(BG)]))
    
                    Bestind = Bestindices.index.values[indbe]
    
                    # print(i)
                    
                elif Indicesname[i] in ['CNN 005', 'CNN 010', 'CNN 020', 'CNN 050', 'CNN 100']:
                    A = Indices_Results_for_Dataset["Value"].values
                    A = A[A != 0]
                    Values = np.array(A.tolist())
                    BestInd = np.argmax(np.sum(Values / np.max(Values, axis=0), axis=1))
                    BestInd = Indices_Results_for_Dataset["Value"].index.values[BestInd]
                    
                    # print(i)
                       
                else:
                    p += 1
                    continue
                    
                    
                ((Bestind < BestExt) * (BestExt < Bestind + num_internos)).astype(int)
                
                    
                Matching.ix[p, :-1] = Matching.ix[p, :-1] + ((Bestind < BestExt) * (BestExt < Bestind + num_internos + num_externos + 2)).astype(int)
                p += 1
        
        MatchingMatriz.append(Matching)
    
    
    
    if len(MatchingMatriz) == 2:
        Matching = MatchingMatriz[0] + MatchingMatriz[1]
        Matching["Index"] = MatchingMatriz[0]["Index"]
        indsortv = np.argsort(-np.sum(Matching[["Kulczynski", "Phi index", "Folkes Mallows", "F Measure", "Jaccard", "Sokal Sneath"]].values, axis=1))
    
        Matching = Matching.ix[indsortv, :].reset_index(drop=True)
        MatchingMatriz[0] = MatchingMatriz[0].ix[indsortv, :].reset_index(drop=True)
        MatchingMatriz[1] = MatchingMatriz[1].ix[indsortv, :].reset_index(drop=True)
            
    else:
        Matching = MatchingMatriz[0]
        indsortv = np.argsort(-np.sum(Matching[["Kulczyski", "Phi index", "Folkes Mallows", "F Measure", "Jaccard", "Sokal Sneath", "Russel Rao", "Entropy"]].values, axis=1))
        Matching = Matching.ix[indsortv, :].reset_index(drop=True)
        MatchingMatriz[0] = MatchingMatriz[0].ix[indsortv, :].reset_index(drop=True)
    
    indsorth = np.argsort(-Matching.values[0, :-1]) 
    Matching = Matching.ix[:, np.append(indsorth, len(Externos))]
    
    plt.figure(100)
    fig = plt.gcf()
    fig.set_size_inches((14, 8), forward=False)
    Image = Matching.values[:, 0:-1].astype(int).T
    
    ax = plt.gca()
    
    im = ax.imshow(Image, cmap="afmhot", vmin=0, vmax=49)
    plt.xticks(range(0, Matching.shape[0]), Matching["Index"].values, rotation='vertical')  
    plt.yticks(range(0, len(Matching.columns) - 1), Matching.columns[:-1]) 
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    maxvalues = np.argmax(Image, axis=1)
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
    for y in range(len(maxvalues)):
        Maxind = Image[y, maxvalues[y]]
        Ind = np.where(Maxind == Image[y, :])[0]
        for a in Ind:
            ax.text(a, y, str(Maxind), fontsize=8, **alignment)
            
    plt.savefig('Results/IndicesResults/' + Algorithm + '/Images/' + m + '/' + Algorithm + '_' + m + '_' + typedata + '_mej.png', bbox_inches='tight') 
    plt.close()
    
    if len(MatchingMatriz) == 2:
        for ext in range(len(Externos)):
            plt.figure(ext)
            width = 0.4 
            plt.bar(np.arange(1, MatchingMatriz[0].shape[0] + 1), MatchingMatriz[0][Externos[ext]], width)
            plt.bar(np.arange(1, MatchingMatriz[1].shape[0] + 1) + width, MatchingMatriz[1][Externos[ext]], width)
            plt.title(Externos[ext] + " Norm: " + m)   
            plt.xticks(range(1, Matching.shape[0] + 1), Matching["Index"].values, rotation='vertical') 
            plt.legend(('Real', 'Simulated'))
            plt.yticks(range(0, int(np.max(np.concatenate((MatchingMatriz[0][Externos[ext]], MatchingMatriz[1][Externos[ext]]))) + 3)))
            fig = plt.gcf()
            fig.set_size_inches((14, 3), forward=False)
            
            plt.savefig('Results/IndicesResults/' + Algorithm + '/Images/' + m + '/' + Externos[ext] + '_' + m + '_' + typedata + '.png', bbox_inches='tight')  
            plt.close()
            
    else:
        for ext in range(len(Externos)):
            plt.figure(ext)
            plt.bar(range(1, Matching.shape[0] + 1), Matching[Externos[ext]])
            plt.title(Externos[ext] + " Norm: " + m)   
            plt.xticks(range(1, Matching.shape[0] + 1), Matching["Index"].values, rotation='vertical') 
            plt.yticks(range(0, int(np.max(Matching[Externos[ext]]) + 3)))
            fig = plt.gcf()
            fig.set_size_inches((14, 3), forward=False)
            
            plt.savefig('Results/IndicesResults/' + Algorithm + '/Images/' + m + '/' + Externos[ext] + '_' + m + '_' + typedata + '.png', bbox_inches='tight')  
            plt.close()