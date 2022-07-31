'''Classification
    Step 1: Training - Input labeled data to a classification algorithm
    Step 2: Prediction - Based on trained algorithm, make a prediction on unseen data
'''
import sklearn.neighbors
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.keys())
# print(digits['DESCR'])
# print(digits['data']) # for training X
#print(digits['images'].shape) #class label

import matplotlib.pyplot as plot
'''Plot image'''
#digit_index = 0
# dig_img = digits.images[0]
# plot.axis('off')
# plot.imshow(dig_img,cmap='binary')
# plot.show()

'''Plot multiple image'''
# fig,axes = plot.subplots(10,10,figsize= (8,8)) #10 row, 10 column, print 100 digit with figsize 8x8
#
# for i, ax in enumerate(axes.flat):
#     dig_img = digits.images[i]
#     ax.axis('off')
#     ax.text(0,0,digits.target[i],color='blue')
#     ax.imshow(dig_img,cmap='binary')
# plot.show()

'''Split Dataset'''
# print(digits.keys())
X,y = digits.data,digits.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

''' Prediction (Suport Vector Machine SVM)'''
# from sklearn.svm import SVC
# svc = SVC(kernel='linear',C=1,random_state=0) # C is a hyperparameter for classification error or margin error, higher c for lower classification error
# svc.fit(X_train,y_train)
# svc_pred = svc.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# accuracy_score = accuracy_score(y_test,svc_pred) # Test set accuracy
# # print(accuracy_score)
#
# from sklearn import metrics
# # print(metrics.classification_report(y_test,svc_pred))
#
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# conf_mat = confusion_matrix(y_test,svc_pred)
# res = sns.heatmap(conf_mat, square=True, annot=True, fmt='d',cmap='Reds')
# plot.show()

''' Prediction Logistic Regrssion'''
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import collections, operator
RunResult = collections.namedtuple('RunResult','model pred accuracy clf_report conf_mat')

# try some model
rand_state =1
model_list = [
    SVC(kernel='linear'),
    KNeighborsClassifier(n_neighbors=5),
    LogisticRegression(max_iter=5000),
    DecisionTreeClassifier(),
    RandomForestClassifier()]

result_list = []

for model in model_list:
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,pred)
    clf_report = metrics.classification_report(y_test,pred)
    conf_mat = confusion_matrix(y_test,pred)
    result_list.append(RunResult(model,pred,accuracy,clf_report,conf_mat))

# sort the results by accuracy
result_list.sort(key=operator.attrgetter('accuracy'),reverse=True)

space = ' '
sns.set(font_scale=0.9)

fig,axes = plot.subplots(2,4,figsize=(15,8))
axes = axes.ravel()

for i, result in enumerate(result_list):
    model_name = type(result.model).__name__
    print(f'{i+1:2}. {model_name:30}        {result.accuracy:6.5f}')
    print('-'*55)
    print(f'    {result.clf_report}')
    sns.heatmap(result.conf_mat,square = True, annot=True, fmt='d', cmap='Reds',\
                ax=axes[i]).set_title(model_name)
    axes[i].set_xlabel('Predict')
    axes[i].set_ylabel('Actual')
plot.show()
best_model = result_list[0].model

''' K-fold validation'''
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# scores = cross_val_score(knn,X_train, y_train, scoring='accuracy', cv=10)
# #print(scores)
#
# import sklearn.metrics
# print(sorted(sklearn.metrics.SCORERS.keys()))

''' Hyperparameter measurement'''
# GridSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
#
# model = KNeighborsClassifier()
# params = [
#     {'n_neighbors':range(2,21),
#     'weights': ['uniform','distance'],}] # 19*2=38 combinations
# search_cv = GridSearchCV(model,params,cv=5,scoring='accuracy',return_train_score=True)
# search_cv.fit(X_train,y_train)
# #print(search_cv.best_estimator_)
# print("GridSearchCV results:\n")
# for i, (mean, params) in enumerate(zip(search_cv.cv_results_['mean_test_score'], search_cv.cv_results_['params'])):
#     if search_cv.best_score_ == mean:
#         print(f' {i+1:2} * {mean:10.5f}, {params}')
#     else:
#         print(f' {i+1:2}   {mean:10.5f}, {params}')

#example 2
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# params = [
#     {'kernel':['linear','rbf','poly','sigmoid']}    # which kernel perform better (with default hyperparameter))
# ]
# model = SVC()
#
# search_cv = GridSearchCV(model,params,cv=5,scoring='accuracy',return_train_score=True)
# search_cv.fit(X_train,y_train)
#
# print(search_cv.best_estimator_)
#
# # print score of each param
# print("GridSearchCV results:\n")
# for i, (mean, params) in enumerate(zip(search_cv.cv_results_['mean_test_score'], search_cv.cv_results_['params'])):
#     if search_cv.best_score_ == mean:
#         print(f' {i+1:2} * {mean:10.5f}, {params}')
#     else:
#         print(f' {i+1:2}   {mean:10.5f}, {params}')

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
params = [
    {'n_neighbors': range(2,21),
    'weights': ['uniform','distance']}]

model = KNeighborsClassifier()
search_rand = RandomizedSearchCV(model,params,cv=5,n_iter=5,scoring='accuracy',return_train_score=True)
search_rand.fit(X_train,y_train)
print(search_rand.best_estimator_)

print("RandomizeSearchCV results:\n")
for i, (mean, params) in enumerate(zip(search_rand.cv_results_['mean_test_score'], search_rand.cv_results_['params'])):
    if search_rand.best_score_ == mean:
        print(f' {i+1:2} * {mean:10.5f}, {params}')
    else:
        print(f' {i+1:2}   {mean:10.5f}, {params}')
























