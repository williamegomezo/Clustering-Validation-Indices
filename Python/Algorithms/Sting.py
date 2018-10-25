import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from math import pi
from skimage import measure

def Sting(data, density, cellsize, minpts):
    
    
            
    # level 1 = root
    # level 2 = 4 squares
    # level 3 = 16 squares
    # level n = 4^(n-1) squares
    
    num_c_p_feat = np.ceil((np.max(data, axis=0) - np.min(data, axis=0)) / cellsize)

    
    size_cell = np.tile(cellsize, (1, data.shape[1]));
    cells_count = (data - np.tile(np.min(data, axis=0), (data.shape[0], 1))) / np.tile(size_cell, (data.shape[0], 1))
    
    # cells in the border are located in lower cells
    ind = np.where(np.floor(cells_count) == cells_count)
    cells_count[ind] = cells_count[ind] - 1
    
    cells_count = np.floor(cells_count) 
    
    cells_with_points = np.vstack({tuple(row) for row in cells_count})
            
#     print(num_c_p_feat)
    
    L = np.zeros(tuple(num_c_p_feat.astype(int)))
    
    for i in range(0, cells_with_points.shape[0]):
        cell_density = np.sum(np.prod(cells_count == cells_with_points[i, :], axis=1))
        
        if cell_density > density:
            L[tuple(cells_with_points[i, :].astype(int))] = 1
    
    
    all_labels = measure.label(L)
    
    recon = np.zeros(data.shape[0])
    for i in range(1, np.max(all_labels) + 1):
        rows = np.array(list(np.where(all_labels == i))).T
        for j in range(0, rows.shape[0]):
            ind = np.where(np.prod(rows[j, :] == cells_count, axis=1))
            recon[ind] = i
    
    p = 0
    for i in range(1, np.max(recon.astype(int)) + 1):
        ind = recon == i
        if np.sum(recon == i) <= minpts:
            recon[ind] = 0
        else:
            p += 1
            recon[ind] = p
            
    return recon
        
        
        
