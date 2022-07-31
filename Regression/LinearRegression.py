'''
    SCIKIT-LEARN
        1. Machine Learning Concepts
            -supervised, unsupervised
        2. Scikit-learn API
        3. model
            regression, classification
    Gold:
        1. Use sk-learn to perform ML tasks on datasrts.
        2. Select the appropriate ML algorithms to train the data, turn the hyperparameters , and eveluate its performances
        3. Pick the model with the best results
'''
'''
    What is AI (Artificial intelligence)?
        1) Computer Vision
        2) EXPERT SYSTEM
        3) FUZZY LOGIC
        4) MACHINE LEARNING
            - Supervised Learning
            - Unsuperised Learning
        5) NATURAL LANGUAGE PROCESSING
        6) ROBOTICS
'''
"""
    Scikit-learn API
        - Accept 2D numpy array/ pandas dataframe
        - Accept numberical values
        - call fit() to learn from data
        - call predict() to predict a label
        - call predict_proba() to predict class probabilities
        - call transform() to modify data
         
    Main Objects
    
        - Estimator - fit()
            - train / learn from data
            - supervised learning
                - estimator.fit(X,y) #X is data, y is label
            - unsupervised learning
                -estimator.fit(X)
                
        - Predictor - predict()
            - make prediction from (unseen) data
                - predictor.predict(data) to predict a label
                - predictor.predict_proba(data) to predict probabilities of each class
            - for supervised learning (and some un-supervised) learning algorithms
        
        - Transformer - fit() & transform()
            - filter or modify the data
            - transformer.transform(data) - must call fit(data) before calling transform(data)
            - transformer.fit_transform(data) - fit and then transform in a single line
        One object can have multiple roles. For example, it could be a estimator and a predictor at the same time.
"""
'''
    # Linear Regression Example
        - Single variabe linear regression
            y = ax+b
            b is intercept
            a is slope
'''
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(1)
x = rng.randint(1,10,50) #from 1-9, gen 50 data
y = 3*x + rng.randn(50) +1 # 1d - price (b =1, x =3) rng.randn(50) from 0-1 gen 50 number follow normal distribution

# plt.scatter(x,y)
# plt.xlabel('# of room')
# plt.ylabel('price (Mil)')
# plt.xticks(range(10))
# plt.xlim([0,10])
# plt.ylim([0,30])
# plt.show()
# print(x.shape)

# find slope and y-intercept
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True) #create model, fit_intercept is a hyperparameter
X = x[:,np.newaxis]
model.fit(X,y) #X is 2d, y is 1d
print(f'intercept: {model.intercept_}') #find y-intercept
print(f'coefficinet: {model.coef_}') #find slope

# step2: prediction
xfit = np.arange(11) # 1 to 10, array
Xfit = xfit[:,np.newaxis] #2d array
yfit = model.predict(Xfit)  #X fit must be a 2d, yfit output 1d
plt.scatter(x,y) #training points

# plt.plot(xfit,yfit)
# plt.xlabel('# of room')
# plt.ylabel(' price (Mil)')
# plt.xlim([0,10]) # x-axis range
# plt.ylim([0,30]) # y-axis range
# plt.show()






