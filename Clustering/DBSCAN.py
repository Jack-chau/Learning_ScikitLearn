'''DBSCAN   # Density-Based Spatial Clustering of Applications with Noise
    - Hyperparameter
        - epsilon radius - all points in this circle form a neightborhood
        - min_sample - if there are more than min_sample points in the neighborhood, it's a core point
    - Each cluster contains at least one core point
'''
'''Generate dataset'''
import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
X1,y1 = make_circles(n_samples=1000, factor=0.4, noise=0.05, random_state=1)
X2,y2 = make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1]],cluster_std=[[0.05]],random_state=0)
X_combine = np.concatenate((X1,X2))
# plot.scatter(X_combine[:,0], X_combine[:,1])
# plot.show()
import numpy as np
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.1, min_samples=5) #eps: epsilon radius, min_sample is minimum sample from a cluster
dbPred = dbscan.fit_predict(X_combine)
# plot.scatter(X_combine[:,0],X_combine[:,1],c=dbPred)
# plot.show()

# CORE POINTS
# print(dbscan.labels_)
# print(np.unique(dbscan.labels_,return_counts=True))
# print(dbscan.core_sample_indices_)
# X_combine[dbscan.core_sample_indices_] # all core point
#dbscan.components_ # all core point

# DBSCAN cannnot predict
# but we can put the result in classification
'''Classification Prediction'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
y_assigned_custer = dbscan.labels_[dbscan.core_sample_indices_]
X = dbscan.components_

knn.fit(X,y_assigned_custer)
print(knn.predict([[-1,-1]]))




































