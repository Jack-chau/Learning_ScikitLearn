'''
    Model parameters Vs Hyperparameters
        - Model parameters (_ at the end)
            - properties that learn from data to describe the model
        - Hyperparameters
            - values to control the learning process
            - you must set the hyperparameter before using ML algorithms
            - e.g. K values for K-means algorithm
    - Generalization in Machine Learning
        -ML is not a memorization process, but a generalization process
        -How well the trained model generalizes to new data (i.e. performance on unseen data prediction)
Main Challenges of Machine Learning
    -Data
        -Not enough data
        -Data is not representative
        -Low quality
        -Missing data, noises
        -incorrect data, irrelevant features, etc.
    -Overfitting
        -Unable to generalize -> overfitting
        -Perform well on training data, but does not generalize well for unseen data
        -Learned from noise data, which is irrelvant
            -Solutions:
                -simpify the model
                -use a fewer parameters
                -use less attributes
                -regularization (constrain the model)
            -Reduce the noise
            -Use more training data
    -Underfitting
        -Selected model is too simple for the data
        -not perform well on both training data and testing data
        -Solutions:
            -Use a more powerful model
            -Reduce constraints on model
'''
'''
Model evaluation
    1. Train & Test on the same data
        A. may overfit the training data, not work well on real world environment
        
    2. Train and Test on different set of data (train_test_split)
        A. Split dataset into training set & testing set. Training on training set, do testing on testing set
        B. sklearn.model_selection.train_test_split()
        
    3. K-Fold Cross-Validation (K-Fold CV)
        A. Split data in to K equal parts for K runs
        B. Select 1 part as validation set and use the remainding parts for training
        C. Repeat for K times
'''
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston() #toy dataset: sklearn Boston housing price
# print(boston.keys())
# print(boston.data)
# print(boston.DESCR)
'''    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town 
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['PRICE']=boston.target
# print(df.head())
# print(boston.data.shape)
# print(boston.target.shape)
# print(df.info())
# print(df.isnull().sum())
# stats
# print(df.describe())
# print(df.PRICE.describe())

# plot dataset
import matplotlib.pyplot as plot
import seaborn as sns

# fig ,ax = plot.subplots(figsize =(10,6))
# sns.boxplot(data=df)
# plot.show()

# df.PRICE.hist()
# plot.title('Boston House Pricing - 1978')
# plot.ylabel('Frequency')
# plot.xlabel('Median Price (#1000)')
# plot.show()
# sns.pairplot(df,x_vars=boston.feature_names,y_vars=['PRICE'],kind='scatter')
# sns.pairplot(df,x_vars=boston.feature_names,y_vars=['PRICE'],kind='kde')
# sns.pairplot(df,kind='reg')
# sns.jointplot(x=df.NOX,y=df.PRICE,kind='reg')
# plot.show()

# Preprocessing
# prepare X and Y
X = df.drop("PRICE",axis=1)
y = boston.target
# Data scaling
# method 1: StandardScaler for standardization (x-mean)/sd
# method 2: MinMaxScaler for min-max normalization (value between 0-1)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#minMaxScaler
minMaxScaler = MinMaxScaler().fit(X)
X_processed = minMaxScaler.transform(X)
df_processed = pd.DataFrame(X_processed,columns=X.columns)
# print(df_processed)
#fig,ax = plot.subplots(1,2,figsize=(10,5))
#sns.boxplot(data=df_processed,ax=ax[0])
#sns.distplot(df_processed['NOX'],ax=ax[1])
# plot.show()
# StandarScaler z-score
X_processed = StandardScaler().fit_transform(X)
df_processed = pd.DataFrame(X_processed,columns=X.columns)
#print(df_processed)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_processed,y,test_size=0.2,random_state=1)

# Training & Evaluation
# Try different Regresssion models
#https://scikit-learn.org/stable/supervised_learning.html
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model_list = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100)]

import collections, operator
RunResult = collections.namedtuple('RunResult','model error pred r2') #RunResult is class, 4 arr

result_list = []
for model in model_list:
    model.fit(X_train, y_train) # train using training set
    pred = model.predict(X_test) # evaluate the performance using test set
    mse = mean_squared_error(y_test, pred) #compute MSE
    r2 = r2_score(y_test, pred) # r2 score (r-square score, max 1, can be -ve)
    rmse = np.sqrt(mse) #root mean square error
    result_list.append(RunResult(model, rmse, pred, r2))

#sort the results by error and then print the result
print(result_list.sort(key=operator.attrgetter('error')))
space=' '
print(f'{space*40}RMSE{space*4}R-square')
for i,result in enumerate (result_list): #i is index enumerate print the index
    print(f'{i+1:2}. {type(result.model).__name__:30}    {result.error:6.2f}     {result.r2:.2f}')

best_model = result_list[0].model
lowest_error = result_list[0].error
best_pred = result_list[0].pred

#performace of the best model
x_index = np.arange(0, pred.shape[0])
dist = y_test - best_pred # compute distance between actual pt and predicted pt


plot.plot(x_index,pred, marker='.', alpha=1, color='b',label = 'Predicted')
plot.plot(x_index,y_test, marker='x', alpha=0.8, color='r',label = 'y_test')
plot.plot(x_index,dist, marker='.', alpha=0.8, color='g',label = 'error')
plot.title(f'{type(best_model).__name__}')
plot.ylabel('Price')
plot.xlabel('Data')
plot.legend()
plot.show()

# Cross-Validation RMSE
from sklearn.model_selection import cross_val_score
score = cross_val_score(best_model,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
#model, x_train, y_train, #score= negative mse, mse lower is better -> nmse higher is better, cross-validation is 5
rmse_cv = np.sqrt(-score)
# print(f'RMSE score: {rmse_cv}')
# print(f'RMSE mean: {rmse_cv.mean():.2f}')
# print(f'RMSE std: {rmse_cv.std():.2f}')

from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
cv_results = cross_validate(model,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
# print(cv_results.keys())
# print('FIT time:' + str(cv_results['fit_time']))
# print('Score time:' + str(cv_results['score_time']))
# rmse_cv = np.sqrt(-cv_results['test_score'])
# print('Test score:' + str(rmse_cv))

result_list = []
for model in model_list:
    scores = cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
    rmse_cv = np.sqrt(-scores)
    result_list.append(RunResult(model,rmse_cv.mean(),None,0))

result_list.sort(key=operator.attrgetter('error'))
space = ' '
print(f'{space*40}RMSE{space*4}')

for i, result in enumerate (result_list):
    print(f'{i+1:2}. {type(result.model).__name__:30}    {result.error:6.2f}')

# Prediction
rows = [[0.02631,0.0,7.5,0.0,0.469,6.420,78.9,4.99,2.1,242.0,17.8,396.90,9.14]]
row_predict = minMaxScaler.transform(rows)
#print(row_predict)

unseen_pred = best_model.predict(row_predict)
print(f'Predicted price: ${unseen_pred[0]*1000:.2f}')