'''Clustering (Unsuperized_learning)'''

'''Generate a dataset'''
import matplotlib.pyplot as plot
from sklearn.datasets import make_blobs
X, label = make_blobs(n_samples=1000, n_features=2,centers=5,random_state=3)

'''Find Cluster'''
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5,random_state=1)
model.fit(X)
#print n_clusters(k) location
# print(f'Cluster centers:\n{model.cluster_centers_}')
# k = model.cluster_centers_
# plot.scatter(X[:,0], X[:,1], c='g', marker='+')
# plot.scatter(k[:,0],k[:,1], color='Red',marker='o')
# plot.show()

'''model.cluster_centers # there are n cluster (numpy ndarray)'''
# print(model.cluster_centers_[model.labels_]) #find each point position
# print(model.predict([[10,10]])) #most clustering algorithm cannot predict new points, but k-means does
# print(k)
# plot.scatter(X[:,0],X[:,1],c=model.labels_, marker='+')
# plot.show()

'''Silhouette'''
# Silhouette_score() compute the mean Silhouette score
# from -1 to +1, -1 => in wrong cluster, 0 => overlapping clusters, +1 => in the right clusters
# from sklearn.metrics import silhouette_score
# score = silhouette_score(X,model.labels_)
# print(score)

'''Find out the best k value using Silhouette Score'''
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# sum_of_square_distances = []
# silhouette_score_list = []
# kRange = range(2,10)
# for k in kRange:
#     kmeans = KMeans(n_clusters=k, random_state=1)
#     kmeans = kmeans.fit(X)
#     # kmeans minimize within-cluster sum-of-squares (inerial)
#     # for this k value
#     # sum_of_squares_distance.append(kmeans.interia)
#     score_avg = silhouette_score(X,kmeans.labels_)
#     silhouette_score_list.append(score_avg)
#
# import matplotlib.pyplot as plot
# plot.plot(kRange,silhouette_score_list,'bx-')
# plot.xlabel('# of cluster (k)')
# plot.ylabel('silhouette score')
# plot.title('Find the Optimal k')
# max_y = max(silhouette_score_list)
# xpos = silhouette_score_list.index(max_y) #index in list
# max_x = kRange[xpos] #index in xlabel (range 2-10)
# plot.plot(max_x, max_y,'ro')
# plot.show()

'''Color Quantization'''
# replacing average color clusters
# k clusters -> k colors
from sklearn.datasets import load_sample_image
img = load_sample_image('flower.jpg')
# print(img.shape) #427 height x 640 width x RGB
# plot.imshow(img)
# plot.show()
# we need a tabel structure for fitting to kmeans
X_color = img.reshape(-1,3) # -1 means doesn't care the shape, 3 is turn to 3 columns
# print(X_color.shape)

'''Color cluster'''
kmeans = KMeans(n_clusters=5).fit(X_color) #find which 5 color have highest representitive
# fancy index to construct new image (using cluster centers as representatice)
segmented_img = kmeans.cluster_centers_[kmeans.labels_] #cluster_centers is the 5 representative color, kernal.labels_ is each pixel or kenel on the img
# kmeans.cluster_centers_ # the average color of each cluster
# the assigned cluster (kenel or pixel) of this data point
# print(segmented_img)
# reshpe segmented_img to original shape
segmented_img = segmented_img.reshape(img.shape)
# show image
# fig, axes = plot.subplots(1,2,figsize= (15,15))
# axes[0].imshow(img)
# axes[1].imshow(segmented_img.astype('uint8'))
# plot.show()
# print(X_color)
'''Color space distribution'''
import numpy as np

#   color scaler since matplotlib color use rgba within 0-1, we have to scale the color from 0-255 to 0-1
def originalColorList(X_color):
    return np.array(X_color)/255

def clusterColorList(cluster_centers_):
    return np.array(cluster_centers_)/255

import matplotlib.pyplot as plot
fig = plot.figure(figsize=plot.figaspect(0.5))
# add_subplot(yxn) #https://www.youtube.com/watch?v=9g_qxtuZmr8
ax1 = fig.add_subplot(1,2,1,projection='3d') #index 1
ax2 = fig.add_subplot(1,2,2,projection = '3d') #index 2
# 3d scatter with original color
ax1.scatter(X_color[:,0],X_color[:,1],X_color[:,2], marker='.',color=originalColorList(X_color))
# 3d scatter with clustered colors
ax2.scatter(X_color[:,0],X_color[:,1],X_color[:,2], marker='.',color=clusterColorList(kmeans.cluster_centers_)[kmeans.labels_])
plot.show()

# print(X_color[0:5])
# array = np.array(X_color)
# print(array[0:5])
# print(array[0:5]/255)




























