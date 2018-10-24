import math
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from blaze.expr.expressions import shape
from scipy import special
from statsmodels.sandbox.distributions.quantize import prob_bv_rectangle
from scipy.spatial.distance import cdist
class CrispIndices:
    def __init__(self, X, labels):
        if len(X) != len(labels):
            raise ValueError('data list and labels must have same length')
        
        self.labels = labels - 1  # As all clusters solutions starts to 1 to number of clusters, minus 1, starts from 0
        self.X = np.array(X)
        
        
        self.n_dims = self.X.shape[1]
        self.N = len(self.X)
        
        
        self.n_clusters = np.max(labels)
        self.M = np.zeros((self.n_clusters, self.n_dims))
        
        self.clusters_len = np.zeros((self.n_clusters,), dtype=np.int)
        for i in range(self.n_clusters):
            self.clusters_len[i] = int(np.sum(self.labels == i))
            self.M[i] = np.mean(X[self.labels == i], axis=0)
        
        self.G = np.mean(X, axis=0)
 

    # разброс внутри кластера k
    # returns float
    def _delta_k(self, k):
        result = 0.0
        for i in range(self.N):
            if self.labels[i] == k:
                result += self.distance(self.X[i], self.M[k])
        result /= self.clusters_len[k]
        return result

    # T
    # returns np.array
    def _total_dispersion_matrix(self):
        return np.cov(self.X.T) * (self.N - 1)

    # WG {k}
    # returns np.array
    def _within_group_k_matrix(self, k):
        if self.clusters_len[k] > 1:
            return np.cov((self.X[self.labels == k]).T) * (np.sum(self.labels == k) - 1)
        else:
            return np.zeros(shape=(self.X.shape[1], self.X.shape[1]))

    # WG
    # returns np.array
    def _within_group_matrix(self):
        WG = np.zeros((self.n_dims, self.n_dims))
        for k in range(self.n_clusters):
            WG += self._within_group_k_matrix(k)
        return WG

    # WGSS{k}
    # returns float
    def _within_cluster_dispersion_k(self, k):
        if self.clusters_len[k] > 1:
            return np.trace(self._within_group_k_matrix(k))
        else:
            return 0

    # WGSS
    # returns float
    def _within_cluster_dispersion(self):
        within = 0;
        for k in range(self.n_clusters):
            within += self._within_cluster_dispersion_k(k)
            
        return within

    # BG
    # returns np.array
    def _between_group_matrix(self):
        return self._total_dispersion_matrix() - self._within_group_matrix()

    # BGSS
    # returns float
    def _between_group_dispersion(self):
        return np.trace(self._between_group_matrix())

    

    def _s_b(self):  # 1621161.
                
        s_b = 0.0
        ndist = 0
        for k1 in range(self.n_clusters):
            ind1 = np.where(self.labels == k1)[0]
            for k2 in range(k1 + 1, self.n_clusters):
                ind2 = np.where(self.labels == k2)[0]
                for i in range(len(ind1)):
                    s_b = s_b + np.sum(np.linalg.norm(self.X[ind1[i]] - self.X[ind2], axis=1))
                    ndist = ndist + self.X[ind2].shape[0]
        
        return s_b

    def _s_w(self):
        """
        :return: float, sum of distances between points in same cluster
        """
        SW = 0
        for k in range(self.n_clusters):
            ind = np.where(self.labels == k)[0]
            for j in range(len(ind) - 1):
                SW = SW + np.sum(np.linalg.norm(self.X[ind[j]] - self.X[ind[j + 1:]], axis=1))
        return SW


    def Int_Ball_Hall(self):
        W = 0
        for i in range(int(self.n_clusters)):
            ind = np.where(self.labels == i)[0]
            W = W + np.sum((self.X[ind, :] - self.M[i]) ** 2) / len(ind)
        return W / int(self.n_clusters)
    
    def Int_Banfeld_Raftery(self):
        index = 0.0
        for k in range(self.n_clusters):
            if self.clusters_len[k] > 1:
                tr = np.trace(self._within_group_k_matrix(k))

                index += self.clusters_len[k] * np.log(tr / self.clusters_len[k])
        return index

    def Int_C_index(self):
        
        SW = 0
        NW = 0 
        Total_distances = list()
        
        for j in range(len(self.X) - 1):
            dists = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1)
            Total_distances.extend(dists)
        
        for k in range(self.n_clusters):
            ind = self.labels == k
            Xtemp = self.X[ind]
            for j in range(len(Xtemp) - 1):
                SW = SW + np.sum(np.linalg.norm(Xtemp[j, :] - Xtemp[j + 1:, :], axis=1));
                
            NW = NW + int(np.sum(ind) * (np.sum(ind) - 1) / 2)
        
        sorted_dists = sorted(Total_distances)
        S_min = sum(sorted_dists[:NW])
        S_max = sum(sorted_dists[-NW:])

        return float((SW - S_min) / (S_max - S_min))

    def Int_Calinski_Harabasz(self):
        traceW = self._within_cluster_dispersion()
        traceB = self._between_group_dispersion()
        return ((self.N - self.n_clusters) * traceB) / ((self.n_clusters - 1) * traceW)
    
    def Int_Davies_Bouldin(self):
        index = 0.0
        
        delta_k = np.zeros((self.n_clusters,))
        for k in range(int(self.n_clusters)):
            ind = np.where(self.labels == k)[0]
            delta_k[k] = np.mean(np.linalg.norm(self.X[ind] - self.M[k], axis=1))

        C = 0
        for k in range(int(self.n_clusters)):
            delta_kk = np.linalg.norm(self.M[k] - np.concatenate((self.M[:k], self.M[k + 1:]), axis=0), axis=1)
            C = C + np.max((delta_k[k] + np.concatenate((delta_k[:k], delta_k[k + 1:]), axis=0)) / delta_kk)
            # np.concatenate((a[:k],a[k+1:]))
        

        return C / self.n_clusters

    def Int_Det_Ratio(self):
        T = self._total_dispersion_matrix()
        WG = self._within_group_matrix()

        detT = np.linalg.det(T)
        detWG = np.linalg.det(WG)
        return detT / detWG
    
    def Int_Dunn11(self):
        return self.Dunn(within=1, between=1)
    
    def Int_Dunn12(self):
        return self.Dunn(within=1, between=2)
    
    def Int_Dunn13(self):
        return self.Dunn(within=1, between=3)
    
    def Int_Dunn14(self):
        return self.Dunn(within=1, between=4)
    
    def Int_Dunn15(self):
        return self.Dunn(within=1, between=5)
    
    def Int_Dunn16(self):
        return self.Dunn(within=1, between=6)
    
    def Int_Dunn21(self):
        return self.Dunn(within=2, between=1)
    
    def Int_Dunn22(self):
        return self.Dunn(within=2, between=2)
    
    def Int_Dunn23(self):
        return self.Dunn(within=2, between=3)
    
    def Int_Dunn24(self):
        return self.Dunn(within=2, between=4)
    
    def Int_Dunn25(self):
        return self.Dunn(within=2, between=5)
    
    def Int_Dunn26(self):
        return self.Dunn(within=2, between=6)
    
    def Int_Dunn31(self):
        return self.Dunn(within=3, between=1)
    
    def Int_Dunn32(self):
        return self.Dunn(within=3, between=2)
    
    def Int_Dunn33(self):
        return self.Dunn(within=3, between=3)
    
    def Int_Dunn34(self):
        return self.Dunn(within=3, between=4)
    
    def Int_Dunn35(self):
        return self.Dunn(within=3, between=5)
    
    def Int_Dunn36(self):
        return self.Dunn(within=3, between=6)
    
    
    def Dunn(self, within, between):
        
        delta_pq = np.empty(shape=(0,))
        delta_CkCk = np.empty(shape=(0,))
        
        if within == 1:
            for i in range(self.n_clusters):
                ind1 = self.labels == i
                for j in np.where(ind1)[0]:
                    delta_CkCk = np.append(delta_CkCk, np.max(np.linalg.norm(self.X[j] - self.X[ind1], axis=1)))
                     
        if within == 2:
            for i in range(self.n_clusters):
                ind1 = self.labels == i
                intrasum = 0
                if len(np.where(ind1)[0]) > 1:
                    for j in np.where(ind1)[0]:
                        intrasum = intrasum + np.sum(np.linalg.norm(self.X[j] - self.X[ind1], axis=1))
                    delta_CkCk = np.append(delta_CkCk, 1 / (self.clusters_len[i] * (self.clusters_len[i] - 1)) * intrasum)
                                                 
        if within == 3:
            for i in range(self.n_clusters):
                ind1 = self.labels == i
                delta_CkCk = np.append(delta_CkCk, 2 / self.clusters_len[i] * np.sum(np.linalg.norm(self.X[ind1] - self.M[i], axis=1)))
                  
        if between == 1:
            for i in range(self.n_clusters):
                ind1 = self.labels == i
                for j in np.where(ind1)[0]:
                    delta_pq = np.append(delta_pq, np.min(np.linalg.norm(self.X[j] - self.X[~ind1], axis=1)))  # distance j to the rest of clusters.
        
        if between == 2:
            for ci in range(self.n_clusters):
                ind1 = self.labels == ci
                for cj in range(ci + 1, self.n_clusters):
                    ind2 = self.labels == cj
                    max_p_q = -float('inf')
                    for j in np.where(ind1)[0]:
                        max_p_q = np.max(np.append(max_p_q, np.max(np.linalg.norm(self.X[j] - self.X[ind2], axis=1)))) 
                    delta_pq = np.append(delta_pq, max_p_q)
        
        if between == 3:
            for ci in range(self.n_clusters):
                ind1 = self.labels == ci
                for cj in range(ci + 1, self.n_clusters):
                    ind2 = self.labels == cj
                    sum_p_q = 0
                    for j in np.where(ind1)[0]:
                        sum_p_q = sum_p_q + np.sum(np.linalg.norm(self.X[j] - self.X[ind2], axis=1)) 
                    delta_pq = np.append(delta_pq, 1 / (self.clusters_len[ci] * self.clusters_len[cj]) * sum_p_q)  
                    
        if between == 4:
            for ci in range(self.n_clusters):
                for cj in range(ci + 1, self.n_clusters):
                    delta_pq = np.append(delta_pq, np.linalg.norm(self.M[ci] - self.M[cj]))  
        
        if between == 5:
            for ci in range(self.n_clusters):
                ind1 = self.labels == ci
                for cj in range(ci + 1, self.n_clusters):
                    ind2 = self.labels == cj
                    sum_pq = np.sum(np.linalg.norm(self.X[ind1] - self.M[cj], axis=1)) + np.sum(np.linalg.norm(self.X[ind2] - self.M[ci], axis=1)) 
                    delta_pq = np.append(delta_pq, 1 / (self.clusters_len[ci] + self.clusters_len[cj]) * sum_pq) 
        
        if between == 6:
            for ci in range(self.n_clusters):
                ind1 = self.labels == ci
                for cj in range(ci + 1, self.n_clusters):
                    ind2 = self.labels == cj
                    
                    max_p = 0
                    for i in np.where(ind1)[0]:
                        max_p = np.max(np.append(max_p, np.min(np.linalg.norm(self.X[i] - self.X[ind2], axis=1)))) 
                        
                    max_q = 0
                    for j in np.where(ind2)[0]:
                        max_q = np.max(np.append(max_q, np.min(np.linalg.norm(self.X[j] - self.X[ind1], axis=1)))) 
                    
                    delta_pq = np.append(delta_pq, np.max(np.append(max_p, max_q)))
                    
        min_delta_pq = np.min(delta_pq)          
        max_delta_CkCk = np.max(delta_CkCk)               
        
        return min_delta_pq / max_delta_CkCk

    def _s_plus_minus(self):
        
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        parity_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
        for j in range(len(self.X)):
            parity_matrix[j, :] = self.labels[j] == self.labels
            
        dist_matrix = dist_matrix + dist_matrix.T    
        np.fill_diagonal(parity_matrix, 5)  # Fill diagonal with any other number   

        S_minus = 0
        S_plus = 0
        for j in range(len(self.X)):
            # print(j)
            ind = (parity_matrix[j, :] == 1)  # Elements within the same cluster
            for i in np.where(ind)[0]:
                indnoparity = parity_matrix == 0
                S_minus = S_minus + np.sum(dist_matrix[i, j] > dist_matrix[indnoparity])
                S_plus = S_plus + np.sum(dist_matrix[i, j] < dist_matrix[indnoparity])
            
        return S_plus / 4, S_minus / 4

    def Int_Baker_Hubert_Gamma(self, s_plus, s_minus):
        return float((s_plus - s_minus) / (s_plus + s_minus))

    def Int_G_Plus(self, s_minus):
        total_points_pairs = self.N * (self.N - 1) / 2
        return float(2 * s_minus / (total_points_pairs * (total_points_pairs - 1)))

    def Int_Ksq_Det_W(self):
        matrix = self._within_group_matrix()
        index = self.n_clusters ** 2
        return index * np.linalg.det(matrix)

    def Int_Log_Det_Ratio(self):
        return self.N * np.log(self.Int_Det_Ratio())

    def Int_Log_SS_Ratio(self):
        bgss = self._between_group_dispersion()
        wgss = self._within_cluster_dispersion()
        return np.log(bgss / wgss)

    def Int_Mcclain_Rao(self):
        SW = self._s_w()
        SB = self._s_b()
        
                
        NW = 0 
        for k in range(self.n_clusters):
            NW = NW + int(self.clusters_len[k] * (self.clusters_len[k] - 1) / 2)
        
        NT = len(self.X) * (len(self.X) - 1) / 2
        NB = NT - NW
        return (NB * SW) / (NW * SB)

    def Int_PBM(self):  # Or I-Index
        
        E1 = np.sum(np.linalg.norm(self.X - self.G, axis=1))
        
        Ek = 0
        Dk = np.empty(shape=(0,))
        for k in range(self.n_clusters):
            ind = self.labels == k
            Ek = Ek + np.sum(np.linalg.norm(self.X[ind] - self.M[k], axis=1))
            
        for k in range(self.n_clusters - 1):
            Dk = np.append(Dk, np.max(np.linalg.norm(self.M[k] - self.M[k + 1:], axis=1)))
        

        return ((E1 * np.max(Dk)) / (self.n_clusters * Ek)) ** 2

    def Int_Point_Biserial(self):
        SW = self._s_w()
        SB = self._s_b()
        
                
        NW = 0 
        for k in range(self.n_clusters):
            NW = NW + int(self.clusters_len[k] * (self.clusters_len[k] - 1) / 2)
        
        NT = len(self.X) * (len(self.X) - 1) / 2
        NB = NT - NW
        
        return float((SW / NW - SB / NB) * math.sqrt(NW * NB) / NT)

    def Int_Ratkowsky_Lance(self):
        bg = self._between_group_matrix()
        ts = self._total_dispersion_matrix()

        p = len(bg)
        r = np.sum(np.diag(bg) / np.diag(ts)) / p
        return math.sqrt(r / self.n_clusters)

    def Int_Ray_Turi(self):
        wgss = self._within_cluster_dispersion()

        deltamin = np.empty(shape=(0,))
        for k in range(self.n_clusters - 1):
            ind = self.labels == k
            deltamin = np.append(deltamin, np.min(np.linalg.norm(self.M[k] - self.M[k + 1:], axis=1) ** 2))
            
        return wgss / (self.N * np.min(deltamin))

    def Int_Scott_Symons(self):
        dets_wg = []

        for k in range(self.n_clusters):
            det = np.linalg.det(self._within_group_k_matrix(k) / self.clusters_len[k])

            if abs(det) < 1e-5:
                return np.nan

            dets_wg.append(det)

        return sum([self.clusters_len[k] * np.log(dets_wg[k]) for k in range(self.n_clusters)])

    def Int_SD(self):
        return self.D_for_sdindex(), self.S_for_sdindex()

    def D_for_sdindex(self):
        Daux = 0
        Dmax = 0
        Dmin = float('Inf')
        for k in range(self.n_clusters):
            mediadiff = np.linalg.norm(np.concatenate((self.M[k] - self.M[:k], self.M[k] - self.M[k + 1:]), axis=0), axis=1)
            Dmax = np.max(np.append(mediadiff, Dmax))
            Dmin = np.min(np.append(mediadiff, Dmin))
            Daux = Daux + 1 / np.sum(mediadiff)
            
        return Dmax / Dmin * Daux

    def S_for_sdindex(self):
        
        Var = list()
        for k in range(self.n_clusters):
            ind = self.labels == k 
            Var.append(np.var(self.X[ind], axis=0))
            
        S = 1 / self.n_clusters * np.sum(np.linalg.norm(Var, axis=1)) / np.linalg.norm(np.var(self.X, axis=0))
    
        return S

    def Int_Sdbw(self):
        
        Var = list()
        for k in range(self.n_clusters):
            ind = self.labels == k 
            if self.clusters_len[k] > 1:
                Var.append(np.var(self.X[ind], axis=0))
                        
        S = 1 / self.n_clusters * np.sum(np.linalg.norm(Var, axis=1)) / np.linalg.norm(np.var(self.X, axis=0))
    
        Sigma = 1 / self.n_clusters * np.sqrt(np.sum(np.linalg.norm(Var, axis=1)))
        
        
        Rkj = 0
        for k in range(self.n_clusters):
            indk = self.labels == k
            for j in range(k + 1, self.n_clusters):
                Hkj = (self.M[k] + self.M[j]) / 2
                indj = self.labels == j
                Y_Hkj = np.sum(np.linalg.norm(self.X[np.logical_or(indk, indj)] - Hkj, axis=1) < Sigma)
                Y_Gk = np.sum(np.linalg.norm(self.X[np.logical_or(indk, indj)] - self.M[k], axis=1) < Sigma)
                Y_Gj = np.sum(np.linalg.norm(self.X[np.logical_or(indk, indj)] - self.M[j], axis=1) < Sigma)
                try:
                    Rkj = Rkj + Y_Hkj / max(Y_Gk, Y_Gj)
                except ZeroDivisionError:
                    return np.nan
                
                
        G = 2 / (self.n_clusters * (self.n_clusters - 1)) * Rkj

        return S + G

    def Int_Silhouette(self):
        return silhouette_score(self.X, self.labels, metric='euclidean')

    def Int_Tau(self, s_plus, s_minus):
        
        NW = 0 
                
        for k in range(self.n_clusters):
            ind = self.labels == k
            NW = NW + int(np.sum(ind) * (np.sum(ind) - 1) / 2)
        
        NT = len(self.X) * (len(self.X) - 1) / 2
        NB = NT - NW
        
        return (s_plus + s_minus) / math.sqrt(NB * NW * NT * (NT - 1) / 2)

    def Int_Trace_W(self):        
        return self._within_cluster_dispersion()

    def Int_Trace_WIB(self):
        wg = self._within_group_matrix()
        bg = self._between_group_matrix()
        return np.matrix.trace(wg.transpose().dot(bg))

    def Int_Wemmert_Gancarski(self):
        
        Jk = 0
        for k in range(self.n_clusters):
            ind = np.where(self.labels == k)[0]
            sumR = 0
            for i in ind:
                diff = np.linalg.norm(self.X[i] - self.M, axis=1)
                ind_cluster = np.zeros(diff.shape[0], dtype=bool)
                ind_cluster[self.labels[i]] = True
                sumR = sumR + diff[ind_cluster] / np.min(diff[~ind_cluster])
            Jk = Jk + float(max(0, self.clusters_len[k] - sumR))

        return Jk / self.N

    def Int_Xie_Beni(self):
        diff = 0
        sumdiff = 0
        for k in range(self.n_clusters):
            ind = np.where(self.labels == k)[0]
            diff = np.linalg.norm(self.X[ind] - self.M[k], axis=1) ** 2
            sumdiff = sumdiff + np.sum(diff)
        
        diff = float('Inf') 
        for k in range(self.n_clusters):
            diff = np.min(np.append(diff, np.linalg.norm(self.M[k] - self.M[k + 1:], axis=1) ** 2))
        
        return sumdiff / (diff * self.N)
    
    def AddedInt_CNN_005(self):
        return self.CNN(5)
    
    def AddedInt_CNN_010(self):
        return self.CNN(10)
    
    def AddedInt_CNN_020(self):
        return self.CNN(20)
    
    def AddedInt_CNN_050(self):
        return self.CNN(50)
    
    def AddedInt_CNN_100(self):
        return self.CNN(100)
    
    def CNN(self, k):
        self.weight = np.zeros(shape=(len(self.X),))
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        parity_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
            
        dist_matrix = dist_matrix + dist_matrix.T    
        np.fill_diagonal(dist_matrix, float('inf'))  # Fill diagonal with any other number
              
        for j in range(len(self.X)):
            parity_matrix[j, :] = self.labels[j] == self.labels
            indicesclosed = np.argsort(dist_matrix[j, :])[0:k]             
            self.weight[j] = np.sum(parity_matrix[j, indicesclosed] == 0) / k 
#             plt.plot(self.X[:,0],self.X[:,1],'.')
#             plt.plot(self.X[j,0],self.X[j,1],'k.')
#             plt.plot(self.X[indicesclosed,0],self.X[indicesclosed,1],'.')
#             plt.title(str(self.weight[j]))
#             plt.show()
        
        separation = np.zeros(shape=(self.n_clusters,))
        compactness = np.zeros(shape=(self.n_clusters,))
        np.fill_diagonal(dist_matrix, 0)
        for c in range(self.n_clusters):
            ind = np.where(self.labels == c)[0]
            separation[c] = np.mean(self.weight[ind])
            
            for j in ind:
                compactness[c] = compactness[c] + np.sum(dist_matrix[j][ind])
            
            compactness[c] = (2 / self.clusters_len[c]) * (self.clusters_len[c] - 1) * compactness[c]
            
        Sep = np.max(separation)    
        Comp = np.sum(compactness)    
        
        return Sep, Comp
    
    def Int_WB_Index(self):
        traceW = self._within_cluster_dispersion()
        traceB = self._between_group_dispersion()
        return self.n_clusters * traceW / traceB 
    
    def Int_WB_Index_zhao(self):
        maxpercluster = np.empty(shape=(0,))
        
        SSB = 0 
        for ci in range(self.n_clusters):
            indi = np.where(self.labels == ci)[0]
            Xaux = self.X[indi]
            if len(indi) > 1:
                maxpercluster = np.append(maxpercluster, np.max(pdist(Xaux, 'euclidean')))
            
            for cj in range(ci + 1, self.n_clusters):
                indj = np.where(self.labels == cj)[0]
                Xaux2 = self.X[np.append(indi, indj)] 
                SSB = SSB + np.min(pdist(Xaux2, 'euclidean'))
            
        SSW = np.max(maxpercluster) + np.sum(self.clusters_len == 1)
        return self.n_clusters * SSW / SSB
    
    def Int_Xu_Index(self):
        traceW = self._within_cluster_dispersion()
        D = self.X.shape[1]
        return np.log10(self.n_clusters * (traceW / (D * self.N ** 2)) ** (D / 2))
    
    def Int_ARsd(self, alpha=0.1):
        
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        parity_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
            
        dist_matrix = dist_matrix + dist_matrix.T    
        
        for j in range(len(self.X)):
            parity_matrix[j, :] = self.labels[j] == self.labels
        
        CMsd = 0
        md = 0 
        for ci in range(self.n_clusters):
            indi = np.where(self.labels == ci)[0]
            sm = 0
            
            if len(indi) > 1:  # If cluster only have 1 element that element does not have radius nor sd
                for i in indi:
                    md = np.max(np.append(sm, np.max(dist_matrix[i, np.where(parity_matrix[i, :])])))
                
                SD = np.sqrt(np.trace(np.cov(self.X[indi].T)))
                CMsd = CMsd + (md - SD) / (md)
            
        
        
        Fn = 0
        k = self.n_clusters
        ci = 0
        labels = self.labels
        X = self.X
        while k > 2:
            indwithin = np.where(labels == ci)[0]
            indbetween = np.where(labels != ci)[0]
            
            dintra = 0
            diff = float('inf')
            for i in indwithin:
                if self.clusters_len[ci] > 1:  # Else dintra 0
                    dintra = dintra + np.min(dist_matrix[i, indwithin])
                
                diff = np.min(np.append(diff, np.min(dist_matrix[i, indbetween])))
            
            dintra = dintra / self.clusters_len[ci]
            dinter = diff
            
            if dinter > 2 * dintra:
                fn = 0
            if dinter <= dintra:
                fn = alpha
            if dintra < dinter and dinter < 2 * dintra:
                fn = alpha * (dinter - dintra) / dintra
            
            Fn = Fn + fn
            labels = np.delete(labels, np.where(labels == ci)[0])
            X = np.delete(X, np.where(labels == ci)[0], axis=0)         
            ci = ci + 1
            k = k - 1
        
        DM = -Fn
        
        return CMsd + DM    
            
    def ComputationsforEC_PC(self):
        
        den_FR = np.sum(self.X, axis=0)
        FR = np.zeros(shape=(self.n_clusters, self.X.shape[1]))
        FP = np.zeros(shape=(self.n_clusters, self.X.shape[1]))
        
        for ci in range(self.n_clusters):
            indi = np.where(self.labels == ci)[0]
            num = np.sum(self.X[indi], axis=0)
            FP[ci, :] = num / np.sum(self.X[indi])
            FR[ci, :] = num / den_FR
        F_measure = 2 * (FR * FP) / (FP + FR)   
        F_measure[np.isnan(F_measure)] = 0  
        F_prom = np.mean(F_measure, axis=0)
        Fall = np.mean(F_prom)
        
        Indtoremove = np.sum(F_measure >= Fall, axis=0) == 0    
        
        F_measureREM = np.delete(F_measure, np.where(Indtoremove), axis=1)
        F_promREM = np.delete(F_prom, np.where(Indtoremove))
        
        firstcondition = F_measureREM > np.tile(F_promREM, (self.n_clusters, 1))
        secondcondition = F_measureREM > Fall
        
        self.selectedfeatures = firstcondition * secondcondition
        
        self.Gi = F_measureREM / np.tile(F_promREM, (F_measureREM.shape[0], 1))
        
        
            
    def Int_EC(self):
        self.ComputationsforEC_PC()
        
        EC = 0
        for ci in range(self.n_clusters):
            activefeat = self.selectedfeatures[ci, :]
            EC = EC + 1 / self.clusters_len[ci] * (np.sum(activefeat == 1) * np.sum(self.Gi[ci, activefeat]) + np.sum(activefeat == 0) * np.sum(1 / self.Gi[ci, activefeat])) / (np.sum(activefeat == 1) + np.sum(activefeat == 0))
        
        return EC / self.n_clusters
    
    def Int_PC(self):
        self.ComputationsforEC_PC()
        
        PC = 0
        for ci in range(self.n_clusters):
            activefeat = self.selectedfeatures[ci, :]
            PC = PC + np.sum(self.Gi[ci, activefeat]) / self.clusters_len[ci]
        
        return PC / self.n_clusters
    
    def Int_BIC(self):
        K = self.n_clusters
        N, M = self.X.shape
        all_variance = 0
        for ci in range(self.n_clusters):
            ind = self.labels == ci
            all_variance = all_variance + np.sum(np.linalg.norm(self.X[ind] - self.M[ci], axis=1) ** 2)
        all_variance = (1.0 / (N - K) / M) * all_variance   
        
        const_term = 0.5 * K * np.log(N) * (M + 1) 
        
        
        
        BIC = self.clusters_len * np.log(self.clusters_len) - self.clusters_len * np.log(N) - ((self.clusters_len * M) / 2) * np.log(2 * np.pi * all_variance) - ((self.clusters_len - 1) * M / 2) 

        return np.sum(BIC) - const_term
    
    
    def Int_AIC(self):
        K = self.n_clusters
        N, M = self.X.shape
        all_variance = 0
        for ci in range(self.n_clusters):
            ind = self.labels == ci
            all_variance = all_variance + np.sum(np.linalg.norm(self.X[ind] - self.M[ci], axis=1) ** 2)
        all_variance = (1.0 / (N - K) / M) * all_variance   
        
        const_term = 0.5 * K * (M + 1) 
        
        AIC = self.clusters_len * np.log(self.clusters_len) - self.clusters_len * np.log(N) - ((self.clusters_len * M) / 2) * np.log(2 * np.pi * all_variance) - ((self.clusters_len - 1) * M / 2) 

        return np.sum(AIC) - const_term
    
    def Int_STR(self):
        return self.Dk_for_STR(), self.Ek_for_STR()
    
    def Dk_for_STR(self):  # 
        
        Dkmax = np.empty(shape=(0,))
        Dkmin = np.empty(shape=(0,))
        Dk = np.empty(shape=(1, 0))
        for k in range(self.n_clusters - 1):
            Dk = np.append(Dk, np.linalg.norm(self.M[k] - self.M[k + 1:], axis=1))
            
        return np.max(Dk) / np.min(Dk)
    
    def Ek_for_STR(self):  #
        
        E1 = np.sum(np.linalg.norm(self.X - self.G, axis=1))
        Ek = 0
        
        for k in range(self.n_clusters):
            ind = self.labels == k
            Ek = Ek + np.sum(np.linalg.norm(self.X[ind] - self.M[k], axis=1))
            
        return E1 / Ek
    
    def Int_Bhargavi_Gowda(self):  # Cut_of f _ratio
        traceW = self._within_cluster_dispersion()
        traceB = self._between_group_dispersion()
        traceT = traceB + traceW
        
        
        Intra_dist = 0
        Inter_dist = 0
        for k in range(self.n_clusters):
            ind = self.labels == k
            Intra_dist = Intra_dist + np.sqrt(np.sum((self.X[ind] - self.M[k]) ** 2))
            
            Inter_dist = Inter_dist + np.sum(np.sqrt(np.sum((self.M[k] - self.M) ** 2, axis=1)))
        
        Inter_dist = Inter_dist / (self.n_clusters) ** 2
        
        return np.abs(traceW * traceT / traceB - Intra_dist / Inter_dist - (self.N - self.n_clusters))
    
    def Int_CS_Measure(self):
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
        dist_matrix = dist_matrix + dist_matrix.T 
        
        Num = 0
        Den = 0
        for k in range(self.n_clusters):
            ind = self.labels == k
            num_aux = 0
            for j in np.where(ind)[0]:
                num_aux = num_aux + np.max(dist_matrix[j, ind])
                
            Num = Num + num_aux / self.clusters_len[k]
            
            Den = Den + np.min(np.linalg.norm(self.M[k] - np.delete(self.M, k, axis=0), axis=1))
            
        return Num / Den
            
    def Int_Score_function(self):
        
        
        BCD = np.sum(np.linalg.norm(self.M - np.mean(self.X, axis=0), axis=1) * self.clusters_len) / (self.N * self.n_clusters)
        WCB = 0
        for k in range(self.n_clusters):
            ind = self.labels == k
            WCB = WCB + np.sum(np.linalg.norm(self.X[ind] - self.M[k], axis=1)) / self.clusters_len[k]
            
        return 1 - 1 / np.exp(np.exp(BCD - WCB))
    
    def Int_Sym(self):
        
        num = np.empty(shape=(0,))
        den = 0
        for k in range(self.n_clusters):
            num = np.append(num, np.linalg.norm(self.M[k] - self.M, axis=1))
            
            ind = np.where(self.labels == k)[0]
            
            if self.clusters_len[k] > 1:
                den = den + np.sum(self.Sym_distance(self.X[ind], self.M[k]))
            
        num = np.max(num)
        
        return num / (self.n_clusters * den)
    
    def Sym_distance(self, X, M):
        
        num_minpts = 2
        
        distances = cdist(2 * M - X, X, 'euclidean')
        indices = np.argsort(distances, axis=1)  # The first 2 distances
        
        Mindistances = np.zeros(shape=(X.shape[0], num_minpts))
        for j in range(indices.shape[0]):
            Mindistances[j, :] = distances[j, indices[j, 0:num_minpts]]
            
        return np.sum(Mindistances, axis=1) / 2
    
    def Int_SymDB(self):
        index = 0.0
        
        delta_k = np.zeros((self.n_clusters,))
        for k in range(int(self.n_clusters)):
            ind = np.where(self.labels == k)[0]
            delta_k[k] = np.mean(self.Sym_distance(self.X[ind], self.M[k]))

        C = 0
        for k in range(int(self.n_clusters)):
            delta_kk = np.linalg.norm(self.M[k] - np.concatenate((self.M[:k], self.M[k + 1:]), axis=0), axis=1)
            C = C + np.max((delta_k[k] + np.concatenate((delta_k[:k], delta_k[k + 1:]), axis=0)) / delta_kk) 
        
        return C / self.n_clusters   
       
    
    def Int_SymD(self):
        
        delta_pq = np.empty(shape=(0,))
        delta_CkCk = np.empty(shape=(0,))
        
        for i in range(self.n_clusters):
            ind1 = self.labels == i
            delta_CkCk = np.append(delta_CkCk, np.max(self.Sym_distance(self.X[ind1], self.M[i])))
            for j in np.where(ind1)[0]:
                delta_pq = np.append(delta_pq, np.min(np.linalg.norm(self.X[j] - self.X[~ind1], axis=1)))  # distance j to the rest of clusters.
        
        min_delta_pq = np.min(delta_pq)          
        max_delta_CkCk = np.max(delta_CkCk)               
        
        return min_delta_pq / max_delta_CkCk
    
    
    def Int_Sym33(self):
        
        delta_pq = np.empty(shape=(0,))
        delta_CkCk = np.empty(shape=(0,))
        
        
        for i in range(self.n_clusters):
            ind1 = self.labels == i
            delta_CkCk = np.append(delta_CkCk, 2 / self.clusters_len[i] * np.sum(self.Sym_distance(self.X[ind1], self.M[i])))
                  
        
        for ci in range(self.n_clusters):
            ind1 = self.labels == ci
            for cj in range(ci + 1, self.n_clusters):
                ind2 = self.labels == cj
                sum_p_q = 0
                for j in np.where(ind1)[0]:
                    sum_p_q = sum_p_q + np.sum(np.linalg.norm(self.X[j] - self.X[ind2], axis=1)) 
                delta_pq = np.append(delta_pq, 1 / (self.clusters_len[ci] * self.clusters_len[cj]) * sum_p_q)  
        
                    
        min_delta_pq = np.min(delta_pq)          
        max_delta_CkCk = np.max(delta_CkCk)               
        
        return min_delta_pq / max_delta_CkCk
            
      
    
    def Int_COP(self):
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
            
        dist_matrix = dist_matrix + dist_matrix.T 
        
        COP = 0
        for k in range(self.n_clusters):
            ind = self.labels == k
            intraCOP = np.sum(np.linalg.norm(self.X[ind] - self.M[k], axis=1))
            den = np.empty(shape=(0,))
            
            indicesn = np.where(~ind)[0]
            
            if len(indicesn) > 0 and np.sum(ind) > 0:
                for j in range(len(indicesn)):
                    den = np.append(den, np.max(dist_matrix[indicesn[j], ind]))
                COP = COP + intraCOP / np.min(den)
    
        return COP / self.N
    
    def Int_SV(self):  # poor formulation
        Num = 0
        Den = 0
        for k in range(self.n_clusters):
            
            Num = Num + np.min(np.linalg.norm(self.M[k] - np.delete(self.M, k, axis=0), axis=1))
            
            ind = self.labels == k
            distances = np.linalg.norm(self.X[ind] - self.M[k], axis=1)
            
            if self.clusters_len[k] > 9:
                Den = Den + 10 / (self.clusters_len[k]) * np.sum(np.sort(distances)[-int(0.1 * self.clusters_len[k]):])  # 10% of the elements in clusters     
    
        try:   
            return Num / Den
        except:
            return np.nan
    
    def Int_OS(self):
        Den = 0
        OV = 0
        for k in range(self.n_clusters):
            indl = self.labels == k
            ind = np.where(indl)[0]
            
            for j in ind:
                a = np.sum(np.linalg.norm(self.X[j] - self.X[indl], axis=1)) / self.clusters_len[k]
                b = np.sum(np.sort(np.linalg.norm(self.X[j] - self.X[~indl], axis=1))[:self.clusters_len[k] + 1]) / self.clusters_len[k]  # sum of ni values closer
                
                if (b - a) / (b + a) < 0.4:
                    ov = a / b
                else:
                    ov = 0  
                OV = OV + ov
                
            distances = np.linalg.norm(self.X[ind] - self.M[k], axis=1)
            if self.clusters_len[k] > 9:
                Den = Den + 10 / (self.clusters_len[k]) * np.sum(np.sort(distances)[-int(0.1 * self.clusters_len[k]):])  # 10% of the elements in clusters      
        try:   
            return OV / Den
        except:
            return np.nan
    
    
    def Int_CVM(self):
        dist_matrix = np.zeros(shape=(len(self.X), len(self.X)))
        
        for j in range(len(self.X) - 1):
            dist_matrix[j, j + 1:] = np.linalg.norm(self.X[j, :] - self.X[j + 1:, :], axis=1);
            
        dist_matrix = dist_matrix + dist_matrix.T 
        
        CVM_first = 0
        CVM_second = 0
        for k in range(self.n_clusters):
            ind = self.labels == k
            dIn = np.empty(shape=(0,))
            Rin_set = np.empty(shape=(0,))
            indicescluster = np.where(ind)[0]
            Dinter = np.empty(shape=(0,))
            
            
            for j in indicescluster:
                dIn = np.append(dIn, np.max(dist_matrix[j, ind]))
                
                indcluster = np.delete(indicescluster, np.where(indicescluster == j))
                
                if self.clusters_len[k] > 1: 
                    Rin_set = np.append(Rin_set, np.min(dist_matrix[j, indcluster])) 
                else:
                    Rin_set = np.append(Rin_set, 1)  # Homogeneus core
                    
                Dinter = np.append(Dinter, np.min(dist_matrix[j, ~ind]))
                
            dIn = np.max(dIn) ** 2
            RIn = np.max(Rin_set) ** 2 / np.mean(Rin_set ** 2)
            CVM_first = CVM_first + dIn / RIn
            CVM_second = CVM_second + np.min(Dinter) ** 2
            
            
            
        return CVM_first * CVM_second
            
            
    def Int_Negentropy_Increment_Biased(self):
        
        first_part = 0
        second_part = -1 / 2 * np.log10(np.linalg.det(np.cov(self.X.T)))
        third_part = 0
        
        for k in range(self.n_clusters):
            ind = self.labels == k
            if self.clusters_len[k] > 1:
                prob = self.clusters_len[k] / self.N
                first_part = first_part + 1 / 2 * prob * np.log10(np.linalg.det(np.cov(self.X[ind].T)))
                third_part = third_part - prob * np.log10(prob)
            
        return first_part + second_part + third_part
            
    def Int_Negentropy_Increment_C(self):  # Clusters with just 1 element are ignore -> Diverges
        first_part = 0
        second_part = -1 / 2 * np.log10(np.linalg.det(np.cov(self.X.T)))
        third_part = 0
        d = self.X.shape[1]
        fourth_part = 0
        
        for k in range(self.n_clusters):
            ind = self.labels == k
            if self.clusters_len[k] > 1:
                prob = self.clusters_len[k] / self.N
                first_part = first_part + 1 / 2 * prob * np.log10(np.linalg.det(np.cov(self.X[ind].T)))
                third_part = third_part - prob * np.log10(prob)
                correctionterm = 0
                
                for j in range(d):
                    correctionterm = correctionterm + special.polygamma(0, (self.clusters_len[k] - (j + 1)) / 2)
                    
                correctionterm = -d * np.log10(2 / (self.clusters_len[k] - 1)) - correctionterm
                
                fourth_part = 1 / 2 * prob * correctionterm
            
        return first_part + second_part + third_part + fourth_part
    
    def Int_Variance_of_Negentropy(self):
        d = self.X.shape[1]
        variance = 0
        
        for k in range(self.n_clusters):
            
            prob = self.clusters_len[k] / self.N
            varianceterm = 0
            if self.clusters_len[k] > 1:
                for j in range(d):
                    varianceterm = varianceterm + special.polygamma(1, (self.clusters_len[k] - (j + 1)) / 2)
            
            if np.isinf(varianceterm):
                return np.nan
            
            variance = variance + 1 / 4 * prob ** 2 * varianceterm
            
        return variance
        
