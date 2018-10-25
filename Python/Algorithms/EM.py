import random
import math
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
          'cyan', 'magenta']



def expectation_maximization(data, num_clusters, iterations):
    # Read data set

    # Select N points random to initiacize the N Clusters
    # initial = [[0.2,0.2],[0.8,0.8]]
    initial = np.array(random.sample(data.tolist(), num_clusters))
#     
    Mindata = np.tile(np.min(data, axis=0), (num_clusters, 1));
    Maxdata = np.tile(np.max(data, axis=0), (num_clusters, 1));
#     
#     initial = (Maxdata - Mindata) * np.random.random_sample((num_clusters, data.shape[1])) + Mindata
    
    # Create N initial Clusters
    clusters_mean = initial
    clusters_std = (Maxdata - Mindata) * np.ones(shape=clusters_mean.shape)
    clusters_prob = 1 / clusters_mean.shape[0] * np.ones((clusters_mean.shape[0],))
    clusters_converge = np.zeros(shape=clusters_mean.shape[0], dtype=bool)
    # Inicialize list of lists to save the new points of cluster
    new_points_cluster = [[] for i in range(num_clusters)]

    converge = False
    it_counter = 0
    while (not converge) and (it_counter < iterations):
        
        recon = np.zeros(shape=(data.shape[0],))
        # Expectation Step
        expectations = np.zeros(shape=(data.shape[0], num_clusters))
        shaped_mean = np.reshape(np.tile(clusters_mean, (1, data.shape[0])), (num_clusters, data.shape[0], data.shape[1]))
        shaped_std = np.reshape(np.tile(clusters_std, (1, data.shape[0])), (num_clusters, data.shape[0], data.shape[1]))
        shaped_data = np.tile(data, (num_clusters, 1, 1))
        shaped_probs = np.tile(np.reshape(clusters_prob, (num_clusters, 1)), (1, data.shape[0]))
        
        
        expectations = np.prod(np.exp(-0.5 * ((shaped_data - shaped_mean) / shaped_std) ** 2) / shaped_std, axis=2) * shaped_probs
        recon = np.argmax(expectations, axis=0)    
        
        oldmean = np.array(clusters_mean)
        
        # Maximization Step
        for i in range(len(clusters_mean)):
            
            points = data[recon == i]
            if points.shape[0] == 0:
                clusters_mean[i, :] = (np.max(data, axis=0) - np.min(data, axis=0)) * np.random.random_sample((data.shape[1],)) - np.min(data, axis=0)
                clusters_std[i, :] = np.mean(Maxdata - Mindata) * np.ones(shape=(1, clusters_mean.shape[1]))
            elif points.shape[0] == 1:
                clusters_mean[i] = points
                clusters_std[i, :] = np.mean(Maxdata - Mindata) * np.ones(shape=(1, clusters_mean.shape[1]))
            else:
                clusters_mean[i] = np.mean(points, axis=0)
                clusters_std[i] = np.std(points, axis=0, ddof=1)
            clusters_prob[i] = np.sum(recon == i) / data.shape[0]
            clusters_converge[i] = np.allclose(oldmean[i], clusters_mean[i])

        # Check that converge all Clusters
        converge = np.prod(clusters_converge)
        
        # Increment counter and delete lists of clusters points
        it_counter += 1

    return recon
