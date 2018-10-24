
import numpy as np

des = 0.0005
mean = [0.2, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate(([x], [y]), axis=0).transpose()
label = 1 * np.ones(x.shape[0])
 
mean = [0.1, 0.9]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 2 * np.ones(x.shape[0])])
 
mean = [0.4, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 3 * np.ones(x.shape[0])])
 
mean = [0.6, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 4 * np.ones(x.shape[0])])
 
mean = [0.8, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 5 * np.ones(x.shape[0])])
 
title = "Well_separated"
np.savetxt('../Datasets/Simulated/' + title + '.txt', np.concatenate((data, label[:, None]), axis=1) , delimiter=' ', fmt=['%.4e', '%.4e', '%d'])



des = 0.0005
mean = [0.2, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate(([x], [y]), axis=0).transpose()
label = 1 * np.ones(x.shape[0])
 
mean = [0.1, 0.9]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 2 * np.ones(x.shape[0])])
 
mean = [0.4, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 3 * np.ones(x.shape[0])])
 
mean = [0.6, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 4 * np.ones(x.shape[0])])
 
mean = [0.8, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 5 * np.ones(x.shape[0])])
 
mean = [0.5, 0.5]
cov = [[des * 100, 0], [0, des * 100]]
x, y = np.random.multivariate_normal(mean, cov, 50).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 6 * np.ones(x.shape[0])])
 
title = "Well_separated noise"
np.savetxt('../Datasets/Simulated/' + title + '.txt', np.concatenate((data, label[:, None]), axis=1) , delimiter=' ', fmt=['%.4e', '%.4e', '%d'])



des = 0.003
mean = [0.5, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
data = np.concatenate(([x], [y]), axis=0).transpose()
label = 1 * np.ones(x.shape[0])
 
mean = [0.3, 0.3]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 50).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 2 * np.ones(x.shape[0])])
 
mean = [0.7, 0.3]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 400).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 3 * np.ones(x.shape[0])])
 
title = "Different Density"
np.savetxt('../Datasets/Simulated/' + title + '.txt', np.concatenate((data, label[:, None]), axis=1) , delimiter=' ', fmt=['%.4e', '%.4e', '%d'])



des = 0.0008
mean = [0.2, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate(([x], [y]), axis=0).transpose()
label = 1 * np.ones(x.shape[0])
 
mean = [0.2, 0.4]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 2 * np.ones(x.shape[0])])
 
mean = [0.65, 0.8]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 3 * np.ones(x.shape[0])])
 
mean = [0.8, 0.8]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 4 * np.ones(x.shape[0])])
 
mean = [0.8, 0.2]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 250).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 5 * np.ones(x.shape[0])])

title = "Sub clusters"
np.savetxt('../Datasets/Simulated/' + title + '.txt', np.concatenate((data, label[:, None]), axis=1) , delimiter=' ', fmt=['%.4e', '%.4e', '%d'])



des = 0.01
mean = [0.5, 0.5]
cov = [[des, 0], [0, des]]
x, y = np.random.multivariate_normal(mean, cov, 1300).T
data = np.concatenate(([x], [y]), axis=0).transpose()
label = 1 * np.ones(x.shape[0])
 
mean = [0.2, 0.2]
cov = [[des / 10, 0], [0, des / 10]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 2 * np.ones(x.shape[0])])
 
mean = [0.8, 0.2]
cov = [[des / 10, 0], [0, des / 10]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
data = np.concatenate([data, np.concatenate(([x], [y]), axis=0).transpose()])
label = np.concatenate([label, 3 * np.ones(x.shape[0])])
 
title = "Skew clusters"
np.savetxt('../Datasets/Simulated/' + title + '.txt', np.concatenate((data, label[:, None]), axis=1) , delimiter=' ', fmt=['%.4e', '%.4e', '%d'])
