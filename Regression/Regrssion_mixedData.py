'''
    Tips dataset
        - Preprocessing for columns with Mixed Types
        - Simpleimputer (remove mising value)
        - OrdinalEncoder (convert ordinal column)
        - OneHotEncoder (convert categorical column/ you can also use pandas.get_dummies(X))
        - pipeline (COmbine several steps)
        - ColumnTransformer (apply transformers or pipelines to differnt columns)
'''
import pandas as pd
import numpy as np
import seaborn as sns ; sns.set
import matplotlib.pyplot as plot

url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/data/csv/tips.csv'
tips = pd.read_csv(url)
# print(tips)
# predice tips
# sns.pairplot(tips,kind='reg')
# sns.jointplot(x=tips.total_bill,y=tips.tip,kind='reg')
# f, ax = plot.subplots(figsize=(6,5))
# sns.boxplot(data=tips)
# plot.show()
# tips.loc[[3,6,9],['sex']] = np.nan
# print(tips.head(11))

'''SimpleImputer to replace missing values'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
tips_imputed = imputer.fit_transform(tips) # return ndarray
tips_imputed = pd.DataFrame(tips_imputed,columns=tips.columns)
# print(tips_imputed.head(10))

'''Text columns to numerical columns
    OneHotEncoder: convert text to number without order # If there are n categories, there will be n new columns
    OrdinalEncoder: convert text to number with order
'''

'''Pandas also support OneHotEncoding feature'''
# tips_pd_encoding = tips.copy()
# tips_pd_encoding = pd.get_dummies(tips_pd_encoding,columns=['sex','smoker','time'],drop_first=True)
# print(tips_pd_encoding)

'''Convert ordinal data to numbers
   Convert nonminal data to columns with numbers
'''

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

tips_raw = tips.copy()
tips_raw.drop('tip',axis=1,inplace=True) # X_train

# for sex column only step1: is null -> most frequent number and turn to OneHotEncoding
sex_pipeline = Pipeline([
        ('null', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary')),])

day_pipeline = Pipeline([
    ('order', OrdinalEncoder(categories=[['Thur','Fri','Sat','Sun']])),
    ('scaler', MinMaxScaler())])

# fit data -> ['column']
preprocess_pipelines = ColumnTransformer([
    ('imputer', sex_pipeline, ['sex']),
    ('onehot', OneHotEncoder(drop='if_binary'), ['smoker','time']),
    ('ordinal', day_pipeline, ['day']),
    ('scaler', MinMaxScaler(), ['total_bill'])],
    remainder='drop') #drop those column haven't mensioned

tips_prepared = preprocess_pipelines.fit_transform(tips_raw)
tips_final_df = pd.DataFrame(tips_prepared)
tips_final_df.columns = ['sex_male','smoker_yes','time_lunch','day','total_bill']
#print(tips_final_df.head(20))

# sns.pairplot(tips_final_df,kind='reg')
# sns.boxplot(data=tips_final_df)
# plot.show()

# split dataset
from sklearn.model_selection import train_test_split
X = tips_final_df
y = tips.tip

X_train,X_test,y_train,y_test = train_test_split(tips_final_df,y,test_size=0.3,random_state=1)

''' Step 2,3: Training and Evaluation
        - Decision Tree
            - Decision tree can be used for both classification and regression problems
            - We will try decision regrssor & random forest regressor for this problem
            - Random forest aggregate multiple decision trees results
'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


model = DecisionTreeRegressor() # max_depth: max. depth of a tree, e.g max_depth =10
model.fit(X_train,y_train) #using training data set for training
pred = model.predict(X_test) # use testing set to evaluvate
mse = mean_squared_error(y_test,pred)
# print('Test Error: \n')
# print(f'MSE: {mse:.4f}')
# print(f'RMSE: {np.sqrt(mse):.4f}') # you can also use cross_val_score()

# Visualize the decision tree
# for random forest, just use model.estimators_[x] to select a tree for visualization
if type(model).__name__ == 'RandomForestRegressor':
    tree = model[0]
else:
    tree = model

# method 1: plot_tree

# fig = plot.figure(figsize=(12,12))
# plot_tree(tree,max_depth=2,filled=True,fontsize=10,feature_names=X_train.columns)
# plot.savefig('tips_decision_tree_plot_tree.png',dpi=100)
# plot.show()
# from graphviz import Source
# #method 2 export_graphviz
# graph = Source(export_graphviz(tree,feature_names=X_train.columns))
# graph.format = 'png'
# graph.render('tis_decision_tree_export_graphviz',view=True)

'''feature importance: total error reduction brought by that feature
# DecisionTreeRegressor & RandomForestRegressor only
# feature_importance_: total mse reduction brought by that feature
'''

# plot feature importances
# sorted_index = np.argsort(model.feature_importances_)[::-1]
# print('Feature importance:')
#
# for i in range(X.shape[1]):
#     print(f'{i+1} - {X_train.columns[sorted_index[i]]:15} {model.feature_importances_[sorted_index[i]]:.5f}')
#
# plot.figure()
# plot.title('Feature importances')
# plot.bar(range(X.shape[1]),model.feature_importances_[sorted_index],color='r',align='center')
# plot.xticks(range(X.shape[1]),X_train.columns[sorted_index])
# plot.show()

# Step 4: Prediction
unseen_row_preprocessed = pd.DataFrame([[19.5,0,'Female','Yes','Fri','Lunch',3]],columns=tips.columns)
unseen_row_preprocessed.drop('tip',inplace=True, axis=1)

# pre-processing is required as well here
# preprocess_pipelines is already fitted, no need to fit again, just transform it
unseen_processed_df = preprocess_pipelines.transform(unseen_row_preprocessed)

# prediction
unseen_pred = model.predict(unseen_processed_df)
print(unseen_row_preprocessed)
print(f'\nPredicted tip: ${unseen_pred[0]:.2f}')
























