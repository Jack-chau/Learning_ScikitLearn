import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot

url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/data/csv/tips.csv'
tips = pd.read_csv(url)
#print(type(tips))
#print(tips.head(10))

# we are goint to predict tips
# sns.pairplot(tips,kind='reg')
# sns.jointplot(x=tips.tip, y=tips.total_bill, kind='reg')
#sns.boxplot(data=tips)
#plot.show()

# similate nan values
tips.loc[[3,6,9],['sex']] = np.nan
#print(tips.head(10))

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
# tips_imputed = imputer.fit_transform(tips)
# tips_imputed = pd.DataFrame(tips_imputed, columns=tips.columns)
#print(tips_imputed)

'''Convert text data to numerical data
    OneHotEncoder = Without order
    OrdinalEncoder = With order
'''

''' OneHotEncoder (pandas)'''
# tips_pd_encoding = tips.copy()
# tips_pd_encoding = pd.get_dummies(tips_pd_encoding,columns=['sex','smoker','time'], drop_first=True)
# print(tips_pd_encoding.head(20))

# Using sklearn for data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,MaxAbsScaler

tip_raw = tips.copy()
# print(tip_raw.head(20))
tip_raw.drop('tip',axis=1,inplace=True) # for X_Train
#print(tip_raw.isna().sum())
sex_pipline = Pipeline([
    ('non_null',SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('oneHot',OneHotEncoder(drop='if_binary'))])

day_pipeline = Pipeline([
    ('order',OrdinalEncoder(categories=[['Thur','Fri','Sat','Sun']])),
    ('scaler',MinMaxScaler())])

preprocess_pipeline = ColumnTransformer([
    ('sex',sex_pipline,['sex']),
    ('OneHotEncoder',OneHotEncoder(drop='if_binary'),['smoker','time']),
    ('day',day_pipeline,['day']),
    ('scaler',MinMaxScaler(),['total_bill'])],
    remainder='drop')
tips_prepared = preprocess_pipeline.fit_transform(tip_raw)
tips_final_df = pd.DataFrame(tips_prepared)
tips_final_df.columns = ['sex_male','smoker_yes','time_lunch','day','total_bill']
# print(tips_final_df)

# split dataset
from sklearn.model_selection import train_test_split
X = tips_final_df
y = tips.tip

X_train,X_test,y_train,y_test = train_test_split(tips_final_df,y,test_size=0.3,random_state=1)

'''Training and Evaluation'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

model = DecisionTreeRegressor()
model.fit(X_train,y_train) # training
pred = model.predict(X_test)
mse = mean_squared_error(y_test,pred)
# print(f'MSE: {mse:.4f}')
# print(f'RMSE: {np.sqrt(mse):.4f}')

# plot:
if type(model).__name__ == 'RandomForestRegressor':
    tree = model[0]
else:
    tree = model
# plot_tree(tree,max_depth=2,filled=True,fontsize=10,feature_names=X_train.columns)
# plot.show()

# Predict

unseen_data = pd.DataFrame([[19.5,0,'Female','Yes','Fri','Lunch',3]],columns=tips.columns)
unseen_data.drop('tip',inplace=True,axis=1)
unseen_data_processed = preprocess_pipeline.transform(unseen_data)
# print(type(unseen_data_processed))
unseen_data_processed_df = pd.DataFrame(unseen_data_processed,columns=tips_final_df.columns)
#print(unseen_data_processed_df)
# Predict
unseen_pred = model.predict(unseen_data_processed_df)
print(unseen_data_processed_df)
print(f'\nPredicted tip: ${unseen_pred[0]:.2f}')





















