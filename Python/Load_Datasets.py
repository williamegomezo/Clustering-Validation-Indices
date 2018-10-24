
import numpy as np
import pandas as pd
from os import listdir

Datasets = pd.DataFrame(columns=['Title', 'Data', 'Labels', 'NumFeat', 'K']);

dirpath = 'Datasets/Simulated/';
dirfiles = listdir(dirpath)

p = 0
for i in dirfiles:
    print(i)
    data = np.loadtxt(dirpath + i);
    label = data[:, 2];
    
    if np.sum(label == 0) > 0:
        data[:, -1] = data[:, -1] + 1
        np.savetxt(dirpath + 'Corrected_' + i, data, delimiter=' ', fmt=['%.4e', '%.4e', '%d'])
    label = data[:, -1]
    
    data = np.delete(data, len(data[0]) - 1, axis=1);
    title = i.split('.')[0];
    
    # Remove feature without info, just for Image dataset maybe
    indextoremove = np.empty((0,))
    for j in range(data.shape[1]):
        if np.sum(np.mean(data[:, j]) == data[:, j]) == data.shape[0]:
            indextoremove = np.append(indextoremove, j)
    data = np.delete(data, indextoremove, axis=1)  
    print("Removed features: ", indextoremove)
    # End removing proccess
    
    # Nan values replaced by the mean of the feature
    for j in range(data.shape[1]):
        indextoreplace = np.isnan(data[:, j])
        if np.sum(indextoreplace) > 0:
            data[indextoreplace, j] = np.mean(data[~indextoreplace, j])
            print("Nan in features: ", j, np.where(indextoreplace))
    
    Datasets.set_value(p, "Title", title);
    Datasets.set_value(p, "Data", data);
    Datasets.set_value(p, "Labels", label);
    Datasets.set_value(p, "NumFeat", data.shape[1]);
    Datasets.set_value(p, "K", np.max(label));
    p += 1

dirpath = 'Datasets/Real/';
dirfiles = listdir(dirpath)
for i in dirfiles:
    print(i)
    
    data = np.loadtxt(dirpath + i)
    label = data[:, -1]
    
    if np.sum(label == 0) > 0:
        data[:, -1] = data[:, -1] + 1
        format = list()
        format = ['%.4e'] * (data.shape[1] - 1)
        format.append('%d')
        np.savetxt(dirpath + i, data, delimiter=' ', fmt=format)   
    label = data[:, -1]
    
    data = np.delete(data, len(data[0]) - 1, axis=1)
    title = i.split('.')[0];
    
    # Remove feature without info, just for Image dataset maybe, and Iono
    indextoremove = np.empty((0,))
    for j in range(data.shape[1]):
        if np.sum(np.mean(data[:, j]) == data[:, j]) == data.shape[0]:
            indextoremove = np.append(indextoremove, j)
    data = np.delete(data, indextoremove, axis=1)  
    print("Removed features: ", indextoremove)
    # End removing proccess
    
    # Nan values replaced by the mean of the feature
    for j in range(data.shape[1]):
        indextoreplace = np.isnan(data[:, j])
        if np.sum(indextoreplace) > 0:
            data[indextoreplace, j] = np.mean(data[~indextoreplace, j])
            print("Nan in features: ", j, np.where(indextoreplace))
    
    Datasets.set_value(p, "Title", title)
    Datasets.set_value(p, "Data", data)
    Datasets.set_value(p, "Labels", label)
    Datasets.set_value(p, "NumFeat", data.shape[1]);
    Datasets.set_value(p, "K", np.max(label) - np.min(label) + 1);
    p += 1
    
    

for i in range(Datasets.shape[0]):
    data = Datasets.loc[i]["Data"]
    label = Datasets.loc[i]["Labels"]
    title = Datasets.loc[i]["Title"]
    print("Dataset: ", title)
    print("# de feaures: ", data.shape[1])
    print("Numero de clusters", np.max(label) - np.min(label) + 1)
    print()
  
  
import pickle
output = open('Datasets/Datasets.pkl', 'wb')
pickle.dump(Datasets, output)

