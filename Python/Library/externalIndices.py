import math
import numpy as np
from nltk.metrics.association import TOTAL
from sklearn import metrics
from matplotlib.mlab import entropy

class ExternalIndices:
    def __init__(self, true_labels, clust_labels):
        
        self.true_labels = true_labels - 1
        self.clust_labels = clust_labels - 1
        self.N = len(self.true_labels)
        self.n_clusters = np.max(clust_labels)
        
    def Ext_Jaccard(self):
        
        a = 0
        b = 0
        c = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            
        return a / (a + b + c)
    
    def Ext_RandStatistic(self):
        
        a = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
        
        M = self.N * (self.N - 1) / 2
        
        return (a + d) / M
    
    def Ext_Folkes_Mallows(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
            
        return a / np.sqrt((a + b) * (a + c))
    
    def Ext_F_Measure(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
            
        return 2 * a / (2 * a + b + c)
    
    def Ext_Hubert_Gamma(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
        
        M = self.N * (self.N - 1) / 2 
        
        return (M * a - (a + b) * (a + c)) / np.sqrt(((a + b) * M - (a + b) ** 2) * ((a + c) * M - (a + c) ** 2))
    
    def Ext_Kulczynski(self):
        a = 0
        b = 0
        c = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
        
        return (1 / 2) * ((a / (a + c)) + (a / (a + b)))
    
    def Ext_McNemar(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
        
        return (d - c) / np.sqrt(d + c)
    
    def Ext_Phi_index(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
        
        return (1 / 2) * ((a / (a + c)) + (a / (a + b)))
    
    def Ext_Rogers_Tanimoto(self):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
            d = d + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:] == False))
        
        return (a + d) / (a + d + 2 * b + 2 * c)
                        
    def Ext_Russel_Rao(self):
        a = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            
        M = self.N * (self.N - 1) / 2 
        return a / M
    
    def Ext_Sokal_Sneath(self):
        a = 0
        b = 0
        c = 0
        for j in range(len(self.true_labels)):
            indtrue = self.true_labels[j] == self.true_labels
            indclus = self.clust_labels[j] == self.clust_labels
            a = a + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:]))  # #pairs of j
            b = b + np.sum(np.logical_and(indtrue[j + 1:], indclus[j + 1:] == False))
            c = c + np.sum(np.logical_and(indtrue[j + 1:] == False, indclus[j + 1:]))
        
        return a / (a + 2 * (b + c))
    
    def Ext_Adjusted_rand_score(self):
        return metrics.adjusted_rand_score(self.true_labels, self.clust_labels)
    
    def Ext_Adjusted_Mutual_Information(self):
        return metrics.adjusted_mutual_info_score(self.true_labels, self.clust_labels)
    
    def Ext_Normalized_Mutual_Information(self):
        return metrics.normalized_mutual_info_score(self.true_labels, self.clust_labels)
    
    def Ext_Mutual_Information(self):
        return metrics.mutual_info_score(self.true_labels, self.clust_labels)
    
    def Ext_Homogeneity_Score(self):
        return metrics.homogeneity_score(self.true_labels, self.clust_labels)
    
    def Ext_Completeness_Score(self):
        return metrics.completeness_score(self.true_labels, self.clust_labels)
    
    def Ext_V_Measure_Score(self):
        return metrics.v_measure_score(self.true_labels, self.clust_labels)
    
    def Ext_Purity(self):
        partition_purity = 0
        for i in range(self.n_clusters):
            elements = self.true_labels[self.clust_labels == i]
            clus_purity = np.max(np.bincount(elements.astype(int)))
            partition_purity = partition_purity + clus_purity
                
        return partition_purity / self.N
    
    def Ext_Conditional_Entropy_V_given_U(self):
        partition_entropy = 0
        
        for i in range(self.n_clusters):
            elements = self.true_labels[self.clust_labels == i]
            prob = np.bincount(elements.astype(int)) / len(elements)
            prob = np.delete(prob, np.where(prob == 0))
            clus_entropy = np.sum(prob * np.log2(prob))
            partition_entropy = partition_entropy + clus_entropy * len(elements)
                
        return partition_entropy / self.N
    
    def Ext_Accuracy1(self):
        
        clusmax = np.empty((0, 2))
        p = 0
        for i in np.unique(self.clust_labels).astype(int):
            elements = self.true_labels[self.clust_labels == i]
            clmax = np.max(np.bincount(elements.astype(int)))
            clargm = np.argmax(np.bincount(elements.astype(int)))
            if p > 0:
                if np.sum(clusmax[:, 0] == clargm) == 0:
                    clusmax = np.concatenate((clusmax, np.array([[clargm, clmax]])), axis=0)
                else:
                    if clusmax[clusmax[:, 0] == clargm, 1] < clmax:
                        clusmax[clusmax[:, 0] == clargm, :] = np.array([[clargm, clmax]])
            else:
                clusmax = np.concatenate((clusmax, np.array([[clargm, clmax]])), axis=0)
            p += 1
        
        
        return np.sum(clusmax[:, 1]) / len(self.clust_labels)
            
        
    
    
    
