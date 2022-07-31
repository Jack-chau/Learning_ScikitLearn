from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.feature_names) 8x8 pixels
# print(digits.data.shape)  (1797x64)
# print(digits.keys())
# print(digits.target_names)

'''Reduce dimension'''
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plot

pca = PCA() #keep all components, so that we can plot graph later

pca.fit(digits.data)
# print(pca.explained_variance_ratio_)
# print(digits.data)
# print(digits.data.shape)
# print(pca.explained_variance_ratio_.shape)

'''Diagram 1'''
# plot.plot(pca.explained_variance_ratio_)
# plot.xlabel('# of compoent')
# plot.ylabel('Explained variance ratio')
# plot.show()

'''Diagram 2''' # the ratio is between 0-1
# plot.plot(np.cumsum(pca.explained_variance_ratio_))
# plot.xlabel('# of component')
# plot.ylabel('Cumulative explained variance')
# plot.show()

'''Preprocessing using PCA'''
# originally there are 8x8 features

'''method 1 (set number of components to keep)'''
# pca1 = PCA(n_components=30) # number of components to keep (the most important one)
# data_transformed = pca1.fit_transform(digits.data)
# print(f'# of column(original): {digits.data.shape}')
# print(f'# of column(after pca1): {data_transformed.shape}')
# # print(pca1.explained_variance_)
# print(sum(pca1.explained_variance_ratio_))

'''method 2 (set ratio)'''
# pca2 = PCA(n_components=0.959)
# data_transform = pca2.fit_transform(digits.data)
# print(f' # of columns (origine) : {digits.data.shape}')
# print(f' # of columns (transformed) : {data_transform.shape}')
# ev_sum = sum(pca2.explained_variance_ratio_)
# print(ev_sum)
#
# '''Classification on low dimensional data'''
# '''Train test split'''
# from sklearn.model_selection import train_test_split
# X_pca, y = data_transform, digits.target
# X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(X_pca, y, random_state=1)
#
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.ensemble import RandomForestClassifier
#
# model = RandomForestClassifier()
# model.fit(X_train_pca,y_train_pca)
# pred = model.predict(X_test_pca)
# accuracy = accuracy_score(y_test_pca,pred)
# clf_report = classification_report(y_test_pca,pred)
# con_mat = confusion_matrix(y_test_pca,pred)
#
# print(clf_report)
# sns.heatmap(con_mat, square=True, annot=True, fmt='d', cmap='Reds')
# plot.show()













