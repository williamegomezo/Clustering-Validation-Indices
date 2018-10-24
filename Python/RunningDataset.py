# Python Packages
import numpy as np
import pandas as pd
import pickle
import os

# Algorithms
from sklearn.cluster import k_means
from Algorithms.EM import expectation_maximization
from Algorithms.Sting import Sting
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from Algorithms.Lamda import lamda

# Standarizations
from Utils.Normalization import minmax_norm
from scipy.stats import zscore 


output = open("Datasets/Datasets.pkl", "rb")
Datasets = pickle.load(output)


for i in range(len(Datasets)):
    print(str(i) + '. ', Datasets.loc[i]["Title"])

indices = np.array(input("Which datasets do you want to compute? Start:Stop \n").split(':')).astype(int)

if len(indices) == 1:
    start = indices[0]
    stop = indices[0]
else:
    start, stop = indices[0], indices[1]

scales = np.array(input("Which scale? 0. No scale 1. Min-Max 2. Zscore \n")).astype(int)

for index in range(start, stop + 1):
    
    data = Datasets.loc[index]["Data"]
    label = Datasets.loc[index]["Labels"]
    title = Datasets.loc[index]["Title"]
    DesiredK = Datasets.loc[index]["K"]
    
    for normi in range(scales, scales + 1):
        if normi == 0:
            Normname = "No norm"
            datan = data
        if normi == 1:
            Normname = "Min-Max"
            datan = minmax_norm(data)
        if normi == 2:
            Normname = "Zscore"
            datan = zscore(data, axis=0)
    
    
    
    
        # Kmeans
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'Kmeans'
        
        Inicio = 1
        if DesiredK < 10:
            Fin = int(np.ceil(3 * np.max(label))) + 1  # Depend on the size, we explore different solutions
        elif DesiredK < 50:
            Fin = int(np.ceil(2 * np.max(label))) + 1
        else:
            Fin = int(np.ceil(1.5 * np.max(label))) + 1  # For example Birch, just explore until 100*1.5 = 150 solutions
               
        for nclusters in range(Inicio, Fin):
               
            if nclusters == 1:
                Iterations = 1  # Same results for k=1
            else:
                Iterations = 10
               
            for i in range(Iterations):
                   
                centroid, recon, inertia = k_means(X=datan, n_clusters=nclusters, init='random')
                   
                print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', nclusters, 'Iteration:', i, 'SSQ:', inertia)
                recon = recon + 1  # Starts in zero
                   
                ListParams = dict()
                ListParams['Iteration'] = i
                   
                row = ClusteringResults.shape[0]
                ClusteringResults.set_value(row, 'Title', title)
                ClusteringResults.set_value(row, 'Norm', Normname)
                ClusteringResults.set_value(row, 'Clusters', max(recon))
                ClusteringResults.set_value(row, 'Recon', recon)
                ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                ClusteringResults.set_value(row, 'Algorithm', Algorithm)
                
                output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                pickle.dump(ClusteringResults, output)
        
        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
        pickle.dump(ClusteringResults, output)
        os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
        
        
        
        
        # EM
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'EM'
        
        Inicio = 1
        if DesiredK < 10:
            Fin = int(np.ceil(3 * np.max(label))) + 1  # Depend on the size, we explore different solutions
        elif DesiredK < 50:
            Fin = int(np.ceil(2 * np.max(label))) + 1
        else:
            Fin = int(np.ceil(1.5 * np.max(label))) + 1  # For example Birch, just explore until 100*1.5 = 150 solutions
               
        for nclusters in range(Inicio, Fin):
               
            if nclusters == 1:
                Iterations = 1  # Same results for k=1
            else:
                Iterations = 10
               
            for i in range(Iterations):
                   
                recon = expectation_maximization(datan, nclusters, 100)
                   
                print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', nclusters, 'Iteration:', i)
                recon = recon + 1  # Starts in zero
                   
                ListParams = dict()
                ListParams['Iteration'] = i
                   
                row = ClusteringResults.shape[0]
                ClusteringResults.set_value(row, 'Title', title)
                ClusteringResults.set_value(row, 'Norm', Normname)
                ClusteringResults.set_value(row, 'Clusters', max(recon))
                ClusteringResults.set_value(row, 'Recon', recon)
                ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                ClusteringResults.set_value(row, 'Algorithm', Algorithm)
        
                output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                pickle.dump(ClusteringResults, output)
        
        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
        pickle.dump(ClusteringResults, output)
        os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
              
              
              
              
              
               
        # DBScan
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'DBScan'
        
        print("Computing range for DBScan: Upper")
        flagminimo = True
        eps = np.mean(np.std(datan, axis=0)) / 10
        alpha = 0.1
        estado = estadoant = 0
        min_samples = 1
        antclusters = 0
        while(flagminimo):
            
            if index in [36, 38]:
                epsmin = np.mean(np.std(datan, axis=0)) / 100
                break
            
            """print(index, title, Normname)
            print(eps, alpha)"""
            
            dbscan_instance = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_instance.fit(datan)
            recon = dbscan_instance.labels_
            nclusters = np.max(recon + 1)
                      
            "print(nclusters, eps, alpha)"
             
            if 2 * DesiredK <= nclusters and nclusters <= 3 * DesiredK:  # For practical issues, just allow clusters below 150
                epsmin = eps
                flagminimo = False
             
            if nclusters == antclusters:
                alpha = alpha * 1.01;
               
            if flagminimo and nclusters < 2 * DesiredK and estado == 0: 
                estado = 1  # Going up
             
            if flagminimo and nclusters > 3 * DesiredK and estado == 1: 
                estado = 0  # Going down
            
            if estado != estadoant:
                alpha = -0.1 * alpha
            
            if alpha <= -1:
                alpha = -0.9
            
            estadoant = estado
            eps = eps + eps * alpha 
            antclusters = nclusters
             
        print("Computing range for DBScan: Lower")
        flagmaximo = True
        eps = epsmin
        alpha = 0.1
        estado = estadoant = 0
        min_samples = 1
        antclusters = 0
        while(flagmaximo):
            
            if index in [36, 38]:
                epsmax = np.mean(np.std(datan, axis=0))
                break
            
            "print(index, title, Normname)"
            dbscan_instance = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_instance.fit(datan)
            recon = dbscan_instance.labels_
            nclusters = np.max(recon + 1)
                      
            "print(nclusters, eps, alpha)"
             
            if 2 <= nclusters and nclusters <= 5:  # For practical issues, just allow clusters below 150
                epsmax = eps
                flagmaximo = False
                 
            if nclusters == antclusters:
                alpha = alpha * 1.01;
                 
            if flagmaximo and nclusters < 2 and estado == 0: 
                estado = 1  # Going up
             
            if flagmaximo and nclusters > 5 and estado == 1: 
                estado = 0  # Going down
                
            if estado != estadoant:
                alpha = -0.1 * alpha
                    
            if alpha <= -1:
                alpha = -0.9
                
            estadoant = estado    
            eps = eps + eps * alpha   
            antclusters = nclusters
        
        print("Finish ranges for DBScan")
        # Now -> Start DBSCAN
         
        for eps in np.linspace(epsmin, epsmax, 10):
               
            for min_samples in range(10):
                   
                dbscan_instance = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_instance.fit(datan)
                   
                recon = dbscan_instance.labels_
                recon[recon >= 0] = recon[recon >= 0] + 1
 
                if(np.max(recon) <= 1):
                    continue
                ListParams = dict()
                ListParams['eps'] = eps
                ListParams['min_samples'] = min_samples
                   
                print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', max(recon))
                row = ClusteringResults.shape[0]
                ClusteringResults.set_value(row, 'Title', title)
                ClusteringResults.set_value(row, 'Norm', Normname)
                ClusteringResults.set_value(row, 'Clusters', max(recon))
                ClusteringResults.set_value(row, 'Recon', recon)
                ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                ClusteringResults.set_value(row, 'Algorithm', Algorithm)

                output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                pickle.dump(ClusteringResults, output)
                
        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
        pickle.dump(ClusteringResults, output)
        os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
        
        
        
        
        # Sting
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'Sting'
        
        if data.shape[1] < 3:
            print("Computing range for Sting: Upper")
            flagminimo = True
            cellsize = np.mean(np.std(datan, axis=0)) / 10
            alpha = 0.1
            estado = estadoant = 0
            minpts = 1
            density = 0 
            antclusters = 0
            while(flagminimo):
                
                if index in [36, 38]:
                    cellsizemin = np.mean(np.std(datan, axis=0)) / 100
                    break
                
#                 print(index, title, Normname)
                recon = Sting(datan, density=density, cellsize=cellsize, minpts=minpts) - 1
                newnclusters = np.max(recon + 1)
                "print(newnclusters, cellsize, alpha)"
                 
                if newnclusters == antclusters:
                    alpha = alpha * 1.01;
                 
                if 2 * DesiredK <= newnclusters and newnclusters <= 3 * DesiredK:  # For practical issues, just allow clusters below 150
#                     print(newnclusters)
                    cellsizemin = cellsize
                    flagminimo = False
                     
                if flagminimo and newnclusters < 2 * DesiredK and estado == 0: 
                    estado = 1  # Going up
                 
                if flagminimo and newnclusters > 3 * DesiredK and estado == 1: 
                    estado = 0  # Going down
                    
                if estado != estadoant:
                    alpha = -0.1 * alpha
                    
                if alpha <= -1:
                    alpha = -0.1
                
                estadoant = estado
                cellsize = cellsize + cellsize * alpha
                antclusters = newnclusters     
                 
                   
            print("Computing range for Sting: Lower")            
            cellsize = cellsizemin
            alpha = 0.1
            flagmaximo = True
            estado = estadoant = 0
            minpts = 1
            density = 1
            antclusters = 0
            while(flagmaximo):
                
                if index in [36, 38]:
                    cellsizemax = np.mean(np.std(datan, axis=0))
                    break
                
#                 print(index, title, Normname)
                recon = Sting(datan, density=density, cellsize=cellsize, minpts=minpts) - 1
                newnclusters = np.max(recon + 1)
#                 print(newnclusters, cellsize, alpha)
                 
                if 2 <= newnclusters and newnclusters <= 5:  # For practical issues, just allow clusters below 150
#                     print(newnclusters)
                    cellsizemax = cellsize
                    flagmaximo = False
                     
                if newnclusters == antclusters:
                    alpha = alpha * 1.01;
                 
                if flagmaximo and newnclusters < 2 and estado == 0: 
                    estado = 1  # Going up
                 
                if flagmaximo and newnclusters > 5 and estado == 1: 
                    estado = 0  # Going down
                    
                if estado != estadoant:
                    alpha = -0.1 * alpha    
                    
                if alpha <= -1:
                    alpha = -0.9
                    
                estadoant = estado
                cellsize = cellsize + cellsize * alpha            
                antclusters = newnclusters
                
            print("Finish ranges for Sting")   
            # Now -> Start STING
             
            for cellsize in np.linspace(cellsizemin, cellsizemax, 10):
                  
                for density in range(5):
                    for minpts in range(5):
                                              
                        recon = Sting(datan, density=density, cellsize=cellsize, minpts=minpts) - 1
                        recon[recon == 0] = -1
                          
                        if(np.max(recon) <= 1):
                            continue
                          
                        ListParams = dict()
                        ListParams['cellsize'] = cellsize
                        ListParams['density'] = density
                        ListParams['minpts'] = minpts
                          
                        print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', max(recon), ListParams)
                        row = ClusteringResults.shape[0]
                        ClusteringResults.set_value(row, 'Title', title)
                        ClusteringResults.set_value(row, 'Norm', Normname)
                        ClusteringResults.set_value(row, 'Clusters', max(recon))
                        ClusteringResults.set_value(row, 'Recon', recon)
                        ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                        ClusteringResults.set_value(row, 'Algorithm', Algorithm)         
                        
                        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                        pickle.dump(ClusteringResults, output)

            output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
            pickle.dump(ClusteringResults, output)
            os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
        
        
        
        
        # Birch 
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'Birch'
        
        Inicio = 2
        if DesiredK < 10:
            Fin = int(np.ceil(3 * np.max(label))) + 1  # Depend on the size, we explore different solutions
        elif DesiredK < 50:
            Fin = int(np.ceil(2 * np.max(label))) + 1
        else:
            Fin = int(np.ceil(1.5 * np.max(label))) + 1  # For example Birch, just explore until 100*1.5 = 150 solutions
               
        new_ls = np.sum(data, axis=0)         
        new_ss = np.dot(new_ls, new_ls)
        new_n = data.shape[0]
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
       
        for nclusters in range(Inicio, Fin):
            for branching_factor in np.linspace(50, np.sqrt(data.shape[0]), 5):
                for threshold in np.linspace(0.1, sq_radius, 5):
                      
                    birch_instance = Birch(branching_factor=int(branching_factor), n_clusters=nclusters, threshold=threshold, compute_labels=True)
                    birch_instance.fit(datan)
                          
                    recon = birch_instance.labels_ + 1
                  
                    if(np.max(recon) <= 1):
                        continue
                      
                    ListParams = dict()
                    ListParams['branching_factor'] = branching_factor
                    ListParams['threshold'] = threshold
                        
                    print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', max(recon))
                    row = ClusteringResults.shape[0]
                    ClusteringResults.set_value(row, 'Title', title)
                    ClusteringResults.set_value(row, 'Norm', Normname)
                    ClusteringResults.set_value(row, 'Clusters', max(recon))
                    ClusteringResults.set_value(row, 'Recon', recon)
                    ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                    ClusteringResults.set_value(row, 'Algorithm', Algorithm) 
                    
                    output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                    pickle.dump(ClusteringResults, output)
 
        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
        pickle.dump(ClusteringResults, output)
        os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
        
        
        
        
        # Lamda
        ClusteringResults = pd.DataFrame(columns=['Title', 'Algorithm', 'Clusters', 'Norm', 'Recon', 'Aditional Params']);
        Algorithm = 'Lamda'
        
        if Normname == "Min-Max":
         
            for mad in ['binomial-centrada', 'binomial', 'gauss']:
                 
                for gad in ['minmax', '3pi']:
                     
                    if gad == '3pi':
                         
                        niv_exi = 1
                         
                        recon = lamda(niv_exi, 5, datan, mad, gad) + 1
 
                        ListParams = dict()
                        ListParams['mad'] = mad
                        ListParams['gad'] = gad      
                        ListParams['niv_exi'] = niv_exi 
                                         
                        print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', max(recon))
                        row = ClusteringResults.shape[0]
                        ClusteringResults.set_value(row, 'Title', title)
                        ClusteringResults.set_value(row, 'Norm', Normname)
                        ClusteringResults.set_value(row, 'Clusters', max(recon))
                        ClusteringResults.set_value(row, 'Recon', recon)
                        ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                        ClusteringResults.set_value(row, 'Algorithm', Algorithm) 
                         
                        output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                        pickle.dump(ClusteringResults, output)
                         
                    else:
                        vectorexi = np.linspace(0, 1, 10)
                         
                        for niv_exi in vectorexi:
                                                     
                             
                            recon = lamda(niv_exi, 5, datan, mad, gad) + 1
                     
                            ListParams = dict()
                            ListParams['mad'] = mad
                            ListParams['gad'] = gad      
                            ListParams['niv_exi'] = niv_exi
                                             
                            print(title, Normname, ', Clusters Labels:', DesiredK, ', Clusters Obt:', max(recon))
                            row = ClusteringResults.shape[0]
                            ClusteringResults.set_value(row, 'Title', title)
                            ClusteringResults.set_value(row, 'Norm', Normname)
                            ClusteringResults.set_value(row, 'Clusters', max(recon))
                            ClusteringResults.set_value(row, 'Recon', recon)
                            ClusteringResults.set_value(row, 'Aditional Params', ListParams)
                            ClusteringResults.set_value(row, 'Algorithm', Algorithm) 
                     
                            output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl', 'wb')
                            pickle.dump(ClusteringResults, output)     
                            
                                
            output = open('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '_FULL.pkl', 'wb')
            pickle.dump(ClusteringResults, output)
            os.remove('Results/ResultsClustering/' + Algorithm + '/' + Algorithm + '_' + title + '_' + Normname + '.pkl')
